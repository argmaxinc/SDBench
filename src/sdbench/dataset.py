# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from pathlib import Path

import numpy as np
import soundfile as sf
from argmaxtools.utils import get_logger
from datasets import Dataset as HfDataset
from datasets import load_dataset
from pyannote.core import Segment, Timeline
from pydantic import BaseModel, Field

from .pipeline_prediction import DiarizationAnnotation, Transcript


logger = get_logger(__name__)


def validate_hf_dataset_schema(ds: HfDataset, expected_columns: list[str]) -> None:
    for col in expected_columns:
        if col not in ds.column_names:
            raise ValueError(f"Dataset is missing expected column: {col}")


class DiarizationDatasetConfig(BaseModel):
    """
    Represents a dataset used for evaluation stored on HuggingFace Hub.

    When working with `diarization` only the dataset should at least have the following columns:
    - `audio`: The audio waveform as Audio object from HuggingFace Datasets.
    - `timestamps_start`: The start timestamps of the diarization segments.
    - `timestamps_end`: The end timestamps of the diarization segments.
    - `speakers`: The speakers of the diarization segments.

    Extra columns are allowed and not directly used, except for:
    - `uem_timestamps`: The UEM of the dataset as a list of tuples `[(start, end), ...]` used for `diarization`
    evaluation.
    - `transcript`: The transcript of the audio as a list of strings (words).
    - `word_speakers`: The speakers of the words in the transcript as a list of strings representing the speaker for
    each word `[speaker, ...]`.
    - `word_timestamps`: The timestamps of the words in the transcript as a list of tuples `[(start, end), ...]`.
    """

    dataset_id: str = Field(..., description="HuggingFace dataset ID. Ex: 'talkbank/callhome'")
    # Not every dataset has a subset or split.
    subset: str | None = Field(
        None,
        description="Subset of the dataset. Ex. 'eng' for the English subset.",
    )
    split: str | None = Field(None, description="Split of the dataset")
    num_samples: int | None = Field(
        None,
        description="Number of samples to take from the dataset. If None, take all samples.",
    )

    def load(self) -> HfDataset:
        ds = load_dataset(self.dataset_id, self.subset, split=self.split)
        if self.num_samples is not None:
            ds = ds.take(self.num_samples)

        return ds


class DiarizationSample(BaseModel):
    audio_name: str = Field(..., description="The name of the audio file")
    waveform: np.ndarray = Field(..., description="The audio waveform as a numpy array with shape (n_samples,)")
    sample_rate: int = Field(..., description="The sample rate of the audio waveform")
    annotation: DiarizationAnnotation = Field(..., description="The annotation of the audio waveform")
    uem: Timeline | None = Field(None, description="The UEM of the audio waveform")
    transcript: Transcript | None = Field(None, description="The transcript of the audio waveform")

    def get_audio_duration(self) -> float:
        return len(self.waveform) / self.sample_rate

    def save_audio(self, output_dir: str | Path) -> Path:
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{self.audio_name}.wav"
        logger.info(f"Saving audio to {output_path}")
        sf.write(output_path, self.waveform, self.sample_rate)

        return output_path

    class Config:
        arbitrary_types_allowed = True


class DiarizationDataset:
    _expected_columns = ["audio", "timestamps_start", "timestamps_end", "speakers"]

    def __init__(self, ds: HfDataset) -> None:
        validate_hf_dataset_schema(ds, self._expected_columns)
        self.ds = ds

    def _prepare_annotation(self, sample: dict) -> DiarizationAnnotation:
        timestamps_start = sample["timestamps_start"]
        timestamps_end = sample["timestamps_end"]
        speakers = sample["speakers"]

        annotation = DiarizationAnnotation()
        for start, end, speaker in zip(timestamps_start, timestamps_end, speakers):
            segment = Segment(start, end)
            annotation[segment] = speaker

        return annotation

    def _prepare_uem(self, sample: dict) -> Timeline | None:
        if "uem_timestamps" not in sample:
            return None

        uem_timestamps = sample["uem_timestamps"]
        uem = Timeline()
        for start, end in uem_timestamps:
            uem.add(Segment(start, end))

        return uem

    def _prepare_transcript(self, sample: dict) -> Transcript:
        if "transcript" not in sample:
            return None

        transcript_words: list[str] = sample["transcript"]

        word_speakers: list[str] | None = sample["word_speakers"] if "word_speakers" in sample else None

        word_timestamps: list[tuple[float, float]] | None = (
            sample["word_timestamps"] if "word_timestamps" in sample else None
        )
        word_timestamps_start = None if word_timestamps is None else [s for s, _ in word_timestamps]
        word_timestamps_end = None if word_timestamps is None else [e for _, e in word_timestamps]

        if word_speakers is not None and len(word_speakers) != len(transcript_words):
            raise ValueError(
                "word_speakers and transcript_words must have the same length, but got "
                f"{len(word_speakers)=} and {len(transcript_words)=} respectively"
            )

        if (
            word_timestamps_start is not None
            and word_timestamps_end is not None
            and len(word_timestamps_start) != len(transcript_words)
        ):
            raise ValueError(
                "word_timestamps_start and word_timestamps_end must have the same length, but got "
                f"{len(word_timestamps_start)=} and {len(transcript_words)=} respectively"
            )

        transcript = Transcript.from_words_info(
            words=transcript_words,
            start=word_timestamps_start,
            end=word_timestamps_end,
            speaker=word_speakers,
        )

        return transcript

    @property
    def dataset_name(self) -> str:
        return self.ds.info.dataset_name

    @property
    def subset(self) -> str:
        return self.ds.config_name

    @property
    def split(self) -> str:
        return self.ds.split

    @property
    def organization(self) -> str:
        # Get the download URL from the chuecksums
        download_url = list(self.ds.info.download_checksums.keys())[0]
        # Extract the organization name from the URL
        parsed_url = download_url.split("hf://datasets/")[-1]
        organization = parsed_url.split("/")[0]

        return organization

    def __str__(self) -> str:
        return f"DiarizationDataset(dataset_id={self.organization}/{self.dataset_name}, subset={self.subset},\
                split={self.split}, num_samples={len(self)})"

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> DiarizationSample:
        row = self.ds[idx]
        annotation = self._prepare_annotation(row)
        uem = self._prepare_uem(row)
        transcript = self._prepare_transcript(row)
        audio = row["audio"]

        audio_name = f"sample_{idx}"
        if "path" in audio and audio["path"] is not None:
            audio_name = Path(audio["path"]).stem

        return DiarizationSample(
            waveform=audio["array"],
            sample_rate=audio["sampling_rate"],
            annotation=annotation,
            uem=uem,
            audio_name=audio_name,
            transcript=transcript,
        )

    @classmethod
    def from_config(cls, config: DiarizationDatasetConfig) -> "DiarizationDataset":
        ds = config.load()
        return cls(ds)


class StreamingSample(BaseModel):
    audio_name: str = Field(..., description="The name of the audio file")
    waveform: np.ndarray = Field(
        ..., description="The audio waveform as a numpy array with shape (n_samples,)"
    )
    sample_rate: int = Field(..., description="The sample rate of the audio waveform")
    annotation: DiarizationAnnotation | None = Field(
        None, description="The annotation of the audio waveform"
    )
    uem: Timeline | None = Field(None, description="The UEM of the audio waveform")
    transcript: Transcript | None = Field(
        None, description="The transcript of the audio waveform"
    )

    def get_audio_duration(self) -> float:
        return len(self.waveform) / self.sample_rate

    def save_audio(self, output_dir: str | Path) -> Path:
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{self.audio_name}.wav"
        logger.info(f"Saving audio to {output_path}")
        sf.write(output_path, self.waveform, self.sample_rate)

        return output_path

    class Config:
        arbitrary_types_allowed = True


class StreamingDataset:
    _expected_columns = ["audio"]

    def __init__(self, ds: HfDataset) -> None:
        # validate_hf_dataset_schema(ds, self._expected_columns)
        self.ds = ds

    def _prepare_annotation(self, sample: dict) -> DiarizationAnnotation:
        timestamps_start = sample["timestamps_start"]
        timestamps_end = sample["timestamps_end"]
        speakers = sample["speakers"]

        annotation = DiarizationAnnotation()
        for start, end, speaker in zip(timestamps_start, timestamps_end, speakers):
            segment = Segment(start, end)
            annotation[segment] = speaker

        return annotation

    def _prepare_transcript(self, sample: dict) -> Transcript:
        transcript_words: list[str] = sample["text"].split(" ")

        word_speakers: list[str] | None = (
            sample["word_speakers"] if "word_speakers" in sample else None
        )

        word_timestamps: list[tuple[float, float]] | None = (
            sample["word_detail"] if "word_detail" in sample else None
        )
        word_timestamps_start = (
            None
            if word_timestamps is None
            else [word["start"] for word in word_timestamps]
        )
        word_timestamps_end = (
            None
            if word_timestamps is None
            else [word["stop"] for word in word_timestamps]
        )

        if word_speakers is not None and len(word_speakers) != len(transcript_words):
            raise ValueError(
                "word_speakers and transcript_words must have the same length, but got "
                f"{len(word_speakers)=} and {len(transcript_words)=} respectively"
            )

        # if (
        #     word_timestamps_start is not None
        #     and word_timestamps_end is not None
        #     and len(word_timestamps_start) != len(transcript_words)
        # ):
        #     raise ValueError(
        #         "word_timestamps_start and word_timestamps_end must have the same length, but got "
        #         f"{len(word_timestamps_start)=} and {len(transcript_words)=} respectively"
        #     )

        transcript = Transcript.from_words_info(
            words=transcript_words,
            start=word_timestamps_start,
            end=word_timestamps_end,
            speaker=word_speakers,
        )

        return transcript

    @property
    def dataset_name(self) -> str:
        return self.ds.info.dataset_name

    @property
    def subset(self) -> str:
        return self.ds.config_name

    @property
    def split(self) -> str:
        return self.ds.split

    @property
    def organization(self) -> str:
        # Get the download URL from the chuecksums
        download_url = list(self.ds.info.download_checksums.keys())[0]
        # Extract the organization name from the URL
        parsed_url = download_url.split("hf://datasets/")[-1]
        organization = parsed_url.split("/")[0]

        return organization

    def __str__(self) -> str:
        return f"DiarizationDataset(dataset_id={self.organization}/{self.dataset_name}, subset={self.subset},\
                split={self.split}, num_samples={len(self)})"

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> StreamingSample:
        row = self.ds[idx]
        transcript = self._prepare_transcript(row)
        audio = row["audio"]

        audio_name = f"sample_{idx}"
        if "path" in audio and audio["path"] is not None:
            audio_name = Path(audio["path"]).stem

        return StreamingSample(
            waveform=audio["array"],
            sample_rate=audio["sampling_rate"],
            uem=None,
            audio_name=audio_name,
            transcript=transcript,
        )

    @classmethod
    def from_config(cls, config: DiarizationDatasetConfig) -> "StreamingDataset":
        ds = config.load()
        return cls(ds)
