# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import os
import subprocess
from pathlib import Path
from typing import Callable, TypedDict

from argmaxtools.utils import get_logger

from ...dataset import DiarizationSample
from ...pipeline_prediction import DiarizationAnnotation
from ..base import Pipeline, PipelineType, register_pipeline
from .common import DiarizationOutput, DiarizationPipelineConfig

__all__ = ["SpeakerKitPipeline", "SpeakerKitPipelineConfig"]

logger = get_logger(__name__)

TEMP_AUDIO_DIR = Path("audio_temp")


class SpeakerKitPipelineConfig(DiarizationPipelineConfig):
    cli_path: str


class SpeakerKitInput(TypedDict):
    audio_path: Path
    output_path: Path
    num_speakers: int | None


class SpeakerKitCli:
    def __init__(self, cli_path: str):
        self.cli_path = cli_path

    def __call__(self, speakerkit_input: SpeakerKitInput) -> Path:
        try:
            cmd = [
                self.cli_path,
                "diarize",
                "--api-key",
                os.environ["SPEAKERKIT_API_KEY"],
                "--audio-path",
                str(speakerkit_input["audio_path"]),
                "--rttm-path",
                str(speakerkit_input["output_path"]),
                "--verbose",
            ]
        except KeyError as e:
            raise ValueError(
                "`SPEAKERKIT_API_KEY` environment variable is not set"
            ) from e

        if speakerkit_input["num_speakers"] is not None:
            cmd.extend(["--num-speakers", str(speakerkit_input["num_speakers"])])

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Diarization CLI stdout:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Diarization CLI failed with error: {e.stderr}") from e

        # Delete the audio file
        speakerkit_input["audio_path"].unlink()

        return speakerkit_input["output_path"]


@register_pipeline
class SpeakerKitPipeline(Pipeline):
    _config_class = SpeakerKitPipelineConfig
    pipeline_type = PipelineType.DIARIZATION

    def build_pipeline(self) -> Callable[[SpeakerKitInput], Path]:
        return SpeakerKitCli(cli_path=self.config.cli_path)

    def parse_input(self, input_sample: DiarizationSample) -> SpeakerKitInput:
        inputs: SpeakerKitInput = {
            "audio_path": input_sample.save_audio(TEMP_AUDIO_DIR),
            "output_path": input_sample.audio_name + ".rttm",
            "num_speakers": None,
        }
        if self.config.use_exact_num_speakers:
            inputs["num_speakers"] = len(set(input_sample.annotation.speakers))

        return inputs

    def parse_output(self, output: Path) -> DiarizationOutput:
        prediction = DiarizationAnnotation.load_annotation_file(output)
        return DiarizationOutput(prediction=prediction)
