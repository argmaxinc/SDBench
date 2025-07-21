# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from typing import Any

from ..pipeline_prediction import Transcript
from .dataset_base import BaseDataset, BaseSample


class TranscriptionSample(BaseSample[Transcript, dict[str, Any]]):
    """Transcription sample for pure transcription tasks."""

    pass


class TranscriptionDataset(BaseDataset[TranscriptionSample]):
    """Dataset for transcription pipelines."""

    _expected_columns = ["audio", "transcript"]
    _sample_class = TranscriptionSample

    def prepare_sample(self, row: dict) -> tuple[Transcript, dict[str, Any]]:
        """Prepare transcript and extra info from dataset row."""
        transcript_words = row["transcript"]
        reference = Transcript.from_words_info(
            words=transcript_words,
            start=row.get("word_timestamps_start"),
            end=row.get("word_timestamps_end"),
            speaker=None,  # No speakers for pure transcription
        )
        extra_info: dict[str, Any] = {}
        return reference, extra_info
