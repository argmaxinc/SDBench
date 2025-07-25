# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from pyannote.core import Annotation
from pyannote.database.util import load_rttm
from pydantic import BaseModel, Field


# Diarization Prediction
class DiarizationAnnotation(Annotation):
    def to_annotation_file(self, output_dir: str, filename: str) -> str:
        path = os.path.join(output_dir, f"{filename}.rttm")
        with open(path, "w") as f:
            self.write_rttm(f)
        return path

    @classmethod
    def from_pyannote_annotation(cls, pyannote_annotation: Annotation) -> "DiarizationAnnotation":
        diarization_annotation = cls(pyannote_annotation.uri)
        for segment, _, speaker in pyannote_annotation.itertracks(yield_label=True):
            diarization_annotation[segment] = speaker
        return diarization_annotation

    @classmethod
    def load_annotation_file(cls, path: str) -> "DiarizationAnnotation":
        pyannote_annotation: dict[str, Annotation] = load_rttm(path)
        if len(pyannote_annotation) != 1:
            raise ValueError(f"Expected exactly one annotation in {path}, but got {len(pyannote_annotation)}")

        pyannote_annotation: Annotation = list(pyannote_annotation.values())[0]
        return cls.from_pyannote_annotation(pyannote_annotation)

    @property
    def timestamps_start(self) -> np.ndarray:
        return np.array([segment.start for segment, _, _ in self.itertracks(yield_label=True)])

    @property
    def timestamps_end(self) -> np.ndarray:
        return np.array([segment.end for segment, _, _ in self.itertracks(yield_label=True)])

    @property
    def speakers(self) -> np.ndarray:
        return np.array([speaker for _, _, speaker in self.itertracks(yield_label=True)])

    @property
    def num_speakers(self) -> int:
        return len(np.unique(self.speakers))


# ASR or Orchestration Prediction
class Word(BaseModel):
    word: str = Field(
        ...,
        description="The word in the transcription",
    )
    start: float | None = Field(
        None,
        description="The start time of the word in seconds",
    )
    end: float | None = Field(
        None,
        description="The end time of the word in seconds",
    )
    speaker: str | None = Field(
        None,
        description="The speaker of the word",
    )

    @classmethod
    def from_string(cls, word: str, speaker: str | None = None) -> "Word":
        return cls(word=word, speaker=speaker)


class Transcript(BaseModel):
    words: list[Word] = Field(
        ...,
        description="The words of the transcript",
    )

    @classmethod
    def from_words_info(
        cls,
        words: list[str],
        start: list[float] | None = None,
        end: list[float] | None = None,
        speaker: list[str] | None = None,
    ) -> "Transcript":
        if start is None:
            start = [None] * len(words)
        if end is None:
            end = [None] * len(words)
        if speaker is None:
            speaker = [None] * len(words)
        return cls(
            words=[
                Word(word=word, start=start, end=end, speaker=speaker)
                for word, start, end, speaker in zip(words, start, end, speaker)
            ]
        )

    def get_words(self) -> list[str]:
        return [word.word for word in self.words]

    def get_speakers(self) -> list[str] | None:
        return [word.speaker for word in self.words] if self.has_speakers else None

    def get_transcript_string(self) -> str:
        "Returns the transcript as a single string of words with spaces between them"
        return " ".join([word.word for word in self.words])

    def get_speakers_string(self) -> str:
        "Returns the speakers as a single string of speakers with spaces between them"
        return " ".join([word.speaker for word in self.words])

    @property
    def has_speakers(self) -> bool:
        return any(word.speaker is not None for word in self.words)

    def to_annotation_file(self, output_dir: str, filename: str) -> str:
        path = os.path.join(output_dir, f"{filename}.csv")
        data = defaultdict(list)
        for word in self.words:
            data["timestamp_start"].append(word.start)
            data["timestamp_end"].append(word.end)
            data["speaker"].append(word.speaker)
            data["word"].append(word.word)

        df = pd.DataFrame(data)
        df.to_csv(path, index=False)
        return path


# NOTE: StreamingTranscript is used only as output of pipelines. The reference for streaming transcript is of type Transcript.
class StreamingTranscript(BaseModel):
    transcript: str = Field(..., description="The final transcript")
    audio_cursor: list[float] | None = Field(None, description="The audio cursor in seconds")
    interim_results: list[str] | None = Field(None, description="The interim results")
    confirmed_audio_cursor: list[float] | None = Field(None, description="The confirmed audio cursor in seconds")
    confirmed_interim_results: list[str] | None = Field(None, description="The confirmed interim results")
    model_timestamps_hypothesis: list[list[dict[str, float]]] | None = Field(
        None,
        description="The model timestamps for the interim results as a list of lists of dictionaries with `start` and `end` keys",
    )
    model_timestamps_confirmed: list[list[dict[str, float]]] | None = Field(
        None,
        description="The model timestamps for the confirmed interim results as a list of lists of dictionaries with `start` and `end` keys",
    )

    def get_words(self) -> list[str]:
        return list(self.transcript.split())

    def get_speakers(self) -> list[str] | None:
        return None

    def to_annotation_file(self, output_dir: str, filename: str) -> str:
        path = os.path.join(output_dir, f"{filename}.json")
        data = {
            "interim_results": self.interim_results,
            "audio_cursor": self.audio_cursor,
            "confirmed_audio_cursor": self.confirmed_audio_cursor,
            "confirmed_interim_results": self.confirmed_interim_results,
            "model_timestamps_hypothesis": self.model_timestamps_hypothesis,
            "model_timestamps_confirmed": self.model_timestamps_confirmed,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        return path
