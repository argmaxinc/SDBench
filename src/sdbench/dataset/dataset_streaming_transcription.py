# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from typing import Any

from ..pipeline_prediction import StreamingTranscript
from .dataset_base import BaseDataset, BaseSample


class StreamingSample(BaseSample[StreamingTranscript, dict[str, Any]]):
    """Streaming transcription sample for real-time transcription tasks."""

    pass


class StreamingDataset(BaseDataset[StreamingSample]):
    """Dataset for streaming transcription pipelines."""

    _expected_columns = ["audio", "text"]
    _sample_class = StreamingSample

    def prepare_sample(self, row: dict) -> tuple[StreamingTranscript, dict[str, Any]]:
        """Prepare streaming transcript and extra info from dataset row."""
        transcript_text = row["text"]
        word_timestamps = row.get("word_detail", [])

        reference = StreamingTranscript(
            transcript=transcript_text,
            audio_cursor=None,
            interim_results=None,
            confirmed_audio_cursor=None,
            confirmed_interim_results=None,
            model_timestamps_hypot=word_timestamps if word_timestamps else None,
            model_timestamps_confirmed=None,
            prediction_time=None,
        )
        extra_info: dict[str, Any] = {}
        return reference, extra_info
