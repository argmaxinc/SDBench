# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import os
import time
from collections import namedtuple
from functools import partial
from pathlib import Path
from typing import Callable

import pvfalcon
from argmaxtools.utils import get_logger
from pyannote.core import Segment

from ...dataset import DiarizationSample
from ...pipeline_prediction import DiarizationAnnotation
from ..base import Pipeline, PipelineType, register_pipeline
from .common import DiarizationOutput, DiarizationPipelineConfig

__all__ = ["PicovoicePipeline", "PicovoicePipelineConfig"]

logger = get_logger(__name__)


TEMP_AUDIO_DIR = Path("audio_temp")
PicovoiceSegment = namedtuple(
    "PicovoiceSegment", ["start_sec", "end_sec", "speaker_tag"]
)


class PicovoicePipelineConfig(DiarizationPipelineConfig):
    pass


def picovoice_diarize(
    audio_path: Path, engine: pvfalcon.Falcon
) -> tuple[list[PicovoiceSegment], str]:
    output = engine.process_file(str(audio_path))
    uri = audio_path.stem
    audio_path.unlink()
    return output, uri


# Picovoice Pipeline expects `PICOVOICE_ACCESS_KEY` to be set with the access key
@register_pipeline
class PicovoicePipeline(Pipeline):
    _config_class = PicovoicePipelineConfig
    pipeline_type = PipelineType.DIARIZATION

    def build_pipeline(self) -> Callable[[Path], tuple[list[PicovoiceSegment], str]]:
        engine = pvfalcon.create(access_key=os.getenv("PICOVOICE_ACCESS_KEY"))

        return partial(picovoice_diarize, engine=engine)

    def parse_input(self, input_sample: DiarizationSample) -> Path:
        return input_sample.save_audio(TEMP_AUDIO_DIR)

    def parse_output(
        self, output: tuple[list[PicovoiceSegment], str]
    ) -> DiarizationOutput:
        segments, uri = output
        prediction = DiarizationAnnotation(uri=uri)
        for segment in segments:
            prediction[
                Segment(segment.start_sec, segment.end_sec)
            ] = segment.speaker_tag
        return DiarizationOutput(prediction=prediction)
