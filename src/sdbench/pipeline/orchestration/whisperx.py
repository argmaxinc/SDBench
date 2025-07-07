# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import gc
from pathlib import Path
from typing import Callable

import numpy as np
import whisperx
from argmaxtools.utils import get_logger
from pydantic import BaseModel, Field

from ...dataset import DiarizationSample
from ...pipeline_prediction import Transcript, Word
from ..base import Pipeline, PipelineConfig, PipelineType, register_pipeline
from .common import OrchestrationOutput


logger = get_logger(__name__)

TEMP_AUDIO_DIR = Path("audio_temp")


class WhisperXPipelineConfig(PipelineConfig):
    model_name: str = Field(
        default="tiny",
        description="The name of the Whisper model to use",
    )
    device: str = Field(
        default="cpu",
        description="The device to run the model on",
    )
    compute_type: str = Field(
        default="float32",
        description="The compute type to use for the model",
    )
    batch_size: int = Field(
        default=16,
        description="The batch size to use when transcribing the audio chunks",
    )


# Creating output schema for WhisperX for better readability
# WordX just extends Word with a score field
class WordX(Word):
    score: float | None

    def to_word(self) -> Word:
        return Word(
            word=self.word,
            start=self.start,
            end=self.end,
            speaker=self.speaker,
        )


class Sentence(BaseModel):
    start: float
    end: float
    text: str
    words: list[WordX]


class WhisperXOutput(BaseModel):
    segments: list[Sentence]
    word_segments: list[WordX]

    def to_words(self) -> list[Word]:
        return [word.to_word() for word in self.word_segments]


class WhisperX:
    def __init__(self, config: WhisperXPipelineConfig):
        self.config = config
        self.model = whisperx.load_model(
            self.config.model_name,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )
        self.diarize_model = whisperx.DiarizationPipeline(device=self.config.device)

    def _align(self, audio: np.ndarray, result: dict) -> dict:
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.config.device)
        result = whisperx.align(
            transcript=result["segments"],
            model=model_a,
            align_model_metadata=metadata,
            audio=audio,
            device=self.config.device,
            return_char_alignments=False,
        )
        del model_a
        gc.collect()
        return result

    def __call__(self, audio_path: Path | str) -> WhisperXOutput:
        if isinstance(audio_path, str):
            audio_path = Path(audio_path)

        # 1. Transcribe with original whisper (batched)
        audio = whisperx.load_audio(str(audio_path))
        result = self.model.transcribe(audio, batch_size=self.config.batch_size)

        # 2. Align whisper output
        result = self._align(audio, result)

        # 3. Diarize Audio
        diarize_segments = self.diarize_model(audio)

        # 4. Assign speaker labels
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # Remove temp audio file
        audio_path.unlink()

        return WhisperXOutput(**result)


@register_pipeline
class WhisperXPipeline(Pipeline):
    _config_class = WhisperXPipelineConfig
    pipeline_type = PipelineType.ORCHESTRATION

    def build_pipeline(self) -> Callable[[Path], WhisperXOutput]:
        return WhisperX(self.config)

    def parse_input(self, input_sample: DiarizationSample) -> Path:
        return input_sample.save_audio(TEMP_AUDIO_DIR)

    def parse_output(self, output: WhisperXOutput) -> OrchestrationOutput:
        prediction = Transcript(
            words=output.to_words(),
        )
        return OrchestrationOutput(
            prediction=prediction,
        )
