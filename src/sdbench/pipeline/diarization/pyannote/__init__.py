# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""PyAnnote pipeline implementation with oracle capabilities."""

from .oracle_diarizer import OracleSpeakerDiarization
from .oracle_inference import OracleSegmenterInference
from .pipeline import *


__all__ = [
    "PyAnnotePipeline",
    "PyAnnotePipelineConfig",
    "OracleSpeakerDiarization",
    "OracleSegmenterInference",
]
