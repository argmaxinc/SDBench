# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from argmaxtools.utils import get_fastest_device, get_logger
from datasets import load_dataset
from huggingface_hub import upload_folder
from pyannote.core import Segment
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from openbench.dataset import DiarizationDataset, DiarizationSample


logger = get_logger(__name__)


def get_overlap_duration(sample: DiarizationSample) -> float:
    return sample.annotation.get_overlap().duration()


def get_total_speech_duration(sample: DiarizationSample) -> float:
    return sample.annotation.get_timeline().support().duration()


class LanguageDetector:
    def __init__(self, model_name: str = "openai/whisper-large-v3", max_duration: float = 30.0):
        self.model_name = model_name
        self.max_duration = max_duration
        self.device = get_fastest_device()
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.processor = WhisperProcessor.from_pretrained(model_name)

    def __call__(self, sample: DiarizationSample) -> str:
        sample_rate = sample.sample_rate
        waveform = sample.waveform
        waveform = waveform[: int(self.max_duration * sample_rate)]
        inputs = self.processor(audio=waveform, sampling_rate=sample_rate, return_tensors="pt").to(self.device)
        language_token = self.model.detect_language(inputs["input_features"])
        return self.processor.decode(language_token).strip("<|").strip("|>")


def compute_speaker_congestion(
    sample: DiarizationSample,
    stride: int = 1,
    window_size: int = 10,
    max_speakers: int = 3,
) -> tuple[float, int, int]:
    """Speaker congestion is defined as the percentage of sliding windows
    of size `window_size` and offset `stride` where the number of active speakers
    is greater than `max_local_speakers`. The higher this number is the harder should be
    the dataset to diarize based on the `max_local_speakers`.
    """
    audio_duration = sample.get_audio_duration()
    padded_audio_duration = math.ceil(audio_duration / stride) * stride
    num_windows = (padded_audio_duration - window_size) // stride - 1
    is_congest = np.zeros(num_windows)
    num_speakers = np.zeros(num_windows)

    for window_idx in range(num_windows):
        window_timestamp_start = window_idx * stride
        window_timestamp_end = window_timestamp_start + window_size
        segment = Segment(window_timestamp_start, window_timestamp_end)
        num_speakers_window = len(sample.annotation.crop(segment).labels())
        num_speakers[window_idx] = num_speakers_window
        is_congest[window_idx] = num_speakers_window > max_speakers

    active_mask = num_speakers > 0
    return (
        is_congest[active_mask].mean(),
        active_mask.sum(),
        np.median(num_speakers[active_mask]),
    )


def get_sample_info(sample: DiarizationSample, language_detector: LanguageDetector | None = None) -> dict[str, float]:
    audio_duration = sample.get_audio_duration()
    overlap_duration = get_overlap_duration(sample)
    total_speech_duration_without_overlap = get_total_speech_duration(sample)
    (
        speaker_congestion,
        num_windows,
        median_num_speakers_per_window,
    ) = compute_speaker_congestion(sample=sample)
    speaker_congestion_stride_2, _, _ = compute_speaker_congestion(sample=sample, stride=2)
    speaker_congestion_stride_4, _, _ = compute_speaker_congestion(sample=sample, stride=4)
    speaker_congestion_stride_5, _, _ = compute_speaker_congestion(sample=sample, stride=5)
    speaker_congestion_stride_10, _, _ = compute_speaker_congestion(sample=sample, stride=10)
    info = {
        "audio_name": sample.audio_name,
        "overlap_duration": overlap_duration,
        "total_speech_duration": total_speech_duration_without_overlap,
        "silence_duration": max(0, audio_duration - total_speech_duration_without_overlap),
        "audio_duration": audio_duration,
        "num_speakers": len(sample.annotation.labels()),
        "speaker_congestion": speaker_congestion,
        "speaker_congestion_windows": num_windows,
        "median_num_speakers_per_window": median_num_speakers_per_window,
        "speaker_congestion_stride_2": speaker_congestion_stride_2,
        "speaker_congestion_stride_4": speaker_congestion_stride_4,
        "speaker_congestion_stride_5": speaker_congestion_stride_5,
        "speaker_congestion_stride_10": speaker_congestion_stride_10,
    }
    if language_detector is not None:
        info["language"] = language_detector(sample)
    return info


def get_dataset_info(dataset: DiarizationDataset, language_detector: LanguageDetector | None = None) -> pd.DataFrame:
    return (
        pd.DataFrame(
            [
                get_sample_info(sample, language_detector)
                for sample in tqdm(dataset, desc="Getting dataset info", total=len(dataset))
            ]
        )
        .reset_index()
        .rename(columns={"index": "sample_id"})
    )


def aggregate_dataset_info(dataset_info: pd.DataFrame) -> pd.DataFrame:
    dataset_agg_info = (
        dataset_info.reset_index()
        .rename(columns={"index": "sample_id"})
        .pipe(lambda df: df.loc[:, ~df.columns.str.contains("_stride")])
        .assign(
            overlap_duration=lambda df: 100 * df["overlap_duration"] / df["audio_duration"].sum(),
            silence_duration=lambda df: 100 * df["silence_duration"] / df["audio_duration"].sum(),
            speaker_congestion=lambda df: 100
            * (df["speaker_congestion"] * df["speaker_congestion_windows"])
            / df["speaker_congestion_windows"].sum(),
            has_congestion=lambda df: 100 * df["speaker_congestion"].gt(0).div(len(df)),
        )
        .pipe(lambda df: df.join(pd.get_dummies(df["language"]).astype(int)))
        .drop(
            columns=[
                "language",
                "sample_id",
                "num_speakers",
                "total_speech_duration",
                "speaker_congestion_windows",
                "median_num_speakers_per_window",
            ]
        )
        .select_dtypes(include=["number"])
        .sum()
        .to_frame()
        .T.assign(
            audio_duration=lambda df: df["audio_duration"].div(3600).round(2),
            min_num_speakers=dataset_info["num_speakers"].min(),
            median_num_speakers=dataset_info["num_speakers"].median().astype(int),
            pct75_num_speakers=dataset_info["num_speakers"].quantile(0.75).astype(int),
            pct90_num_speakers=dataset_info["num_speakers"].quantile(0.9).astype(int),
            max_num_speakers=dataset_info["num_speakers"].max().astype(int),
        )
        .rename(
            columns={
                "audio_duration": "Total Duration (h)",
                "overlap_duration": "% Overlap",
                "silence_duration": "% Silence",
                "speaker_congestion": "% Speaker Congestion",
                "has_congestion": "% Congested Samples",
                "min_num_speakers": "Min Speakers",
                "median_num_speakers": "Median Speakers",
                "pct75_num_speakers": "75% Speakers",
                "pct90_num_speakers": "90% Speakers",
                "max_num_speakers": "Max Speakers",
            }
        )
    )
    first_cols = [
        "Total Duration (h)",
        "% Overlap",
        "% Silence",
        "Min Speakers",
        "Median Speakers",
        "75% Speakers",
        "90% Speakers",
        "Max Speakers",
    ]
    language_cols = [col for col in dataset_agg_info.columns if col not in first_cols]
    return dataset_agg_info[first_cols + language_cols]


def main(
    dataset_id: str,
    subset: str | None = None,
    output_dir: str = "dataset_statistics",
    repo_id: str = "argmaxinc/interspeech-artifacts",
) -> None:
    output_dir = Path(output_dir)
    agg_data_dir = output_dir / "agg_info"
    per_sample_data_dir = output_dir / "per_sample_info"

    # Hacky way to handle no subset when running the workflow to get the statistics
    if subset is not None and subset == "None":
        subset = None

    agg_data_dir.mkdir(parents=True, exist_ok=True)
    per_sample_data_dir.mkdir(parents=True, exist_ok=True)

    # Load raw dataset to get available splits
    raw_dataset = load_dataset(dataset_id, subset)
    splits = raw_dataset.keys()
    logger.info(f"Processing splits: {splits}")

    language_detector = LanguageDetector()

    for split in splits:
        try:
            logger.info(f"Processing split: {split}")
            dataset = DiarizationDataset(raw_dataset[split])
            logger.info(f"Dataset loaded with {len(dataset)} samples")

            dataset_info = get_dataset_info(dataset, language_detector)
            logger.info(f"Dataset info collected with {len(dataset_info)} samples")
            dataset_agg_info = aggregate_dataset_info(dataset_info)
            logger.info(f"Dataset agg info collected with {len(dataset_agg_info)} samples")

            prefix = f"{dataset_id.replace('/', '__')}_{subset}_{split}"
            logger.info(f"Saving dataset agg info to {agg_data_dir / f'{prefix}_agg_info.csv'}")
            dataset_agg_info.to_csv(agg_data_dir / f"{prefix}_agg_info.csv", index=False)
            logger.info(f"Saving dataset per sample info to {per_sample_data_dir / f'{prefix}_per_sample_info.csv'}")
            dataset_info.to_csv(per_sample_data_dir / f"{prefix}_per_sample_info.csv", index=False)
        except Exception as e:
            logger.error(f"Error processing split {split}: {e}")
            continue

    logger.info(f"Pushing dataset statistics to {repo_id}")
    upload_folder(repo_id=repo_id, folder_path=str(output_dir), path_in_repo=str(output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-id", type=str, required=True)
    parser.add_argument("--subset", type=str, required=False, default=None)
    parser.add_argument("--output-dir", type=str, required=False, default="dataset_statistics")
    parser.add_argument("--repo-id", type=str, required=False, default="argmaxinc/interspeech-artifacts")

    args = parser.parse_args()
    main(
        dataset_id=args.dataset_id,
        subset=args.subset,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
    )
