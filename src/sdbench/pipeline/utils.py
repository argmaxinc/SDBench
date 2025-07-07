# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Shared utilities for pipeline implementations."""

from enum import Enum, auto
from typing import Protocol, runtime_checkable


class PipelineType(Enum):
    DIARIZATION = auto()
    TRANSCRIPTION = auto()
    ORCHESTRATION = auto()
    STREAMING_TRANSCRIPTION = auto()


# All prediction classes that we output should conform to this 
# TODO: add `load_from_annotation` to protocol
@runtime_checkable
class PredictionProtocol(Protocol):
    def to_annotation_file(self, output_dir: str, filename: str) -> str:
        """
        Must implement a method to save the prediction to a file.

        Args:
            output_dir: The directory to save the prediction to.
            filename: The filename to save the prediction to without the extension

        Returns:
            The path to the saved prediction.
        """
        pass
