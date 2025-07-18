# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from pydantic import Field

from ...pipeline_prediction import Transcript
from ..base import PipelineConfig, PipelineOutput


# TODO: Add support for forced language transcription
class TranscriptionConfig(PipelineConfig):
    force_language: bool = Field(
        False, description="Force the language of the audio files i.e. hint the model to use the correct language."
    )


class TranscriptionOutput(PipelineOutput[Transcript]):
    pass
