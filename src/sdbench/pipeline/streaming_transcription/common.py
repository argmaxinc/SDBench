# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from ...pipeline_prediction import StreamingTranscript
from ..base import PipelineConfig, PipelineOutput


class StreamingTranscriptionConfig(PipelineConfig):
    endpoint_url: str


class StreamingTranscriptionOutput(PipelineOutput[StreamingTranscript]):
    ...
