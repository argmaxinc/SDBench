# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import time
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Callable, Generic, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, Field

from ..dataset import DiarizationSample


ParsedInput = TypeVar("ParsedInput")
GenericOutput = TypeVar("GenericOutput")

PIPELINE_REGISTRY: dict[str, "Pipeline"] = {}


class PipelineType(Enum):
    DIARIZATION = auto()
    TRANSCRIPTION = auto()
    ORCHESTRATION = auto()


def register_pipeline(cls: type["Pipeline"]) -> type["Pipeline"]:
    PIPELINE_REGISTRY[cls.__name__] = cls
    return cls


class PipelineConfig(BaseModel):
    out_dir: str = "."
    # If this variable is set to some value (n), the benchmark runner will split the work
    # across a pool of n processes. Otherwise, it will run the benchmark sequentially.
    num_worker_processes: int = Field(None, description="Number of worker processes to use for parallel processing")

    per_worker_chunk_size: int = Field(1, description="Number of samples to process in each worker at a time")


# All prediction classes that we output should conform to this protocol
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


Prediction = TypeVar("Prediction", bound=PredictionProtocol)


class PipelineOutput(BaseModel, Generic[Prediction]):
    prediction: Prediction = Field(..., description="Pipeline final prediction")
    prediction_time: float | None = Field(None, description="The time taken to perform the prediction")

    class Config:
        arbitrary_types_allowed = True


class Pipeline(ABC):
    """
    Abstract class for diarization pipelines.
    """

    _config_class: PipelineConfig
    pipeline_type: PipelineType

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.pipeline = self.build_pipeline()

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if not hasattr(cls, "_config_class"):
            raise ValueError(f"Pipeline {cls.__name__} must define a _config_class attribute")
        if not hasattr(cls, "pipeline_type"):
            raise ValueError(f"Pipeline {cls.__name__} must define a pipeline_type attribute")

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "Pipeline":
        config_dict = config["config"] if "config" in config else config
        config = cls._config_class(**config_dict)
        return cls(config)

    @abstractmethod
    def build_pipeline(self) -> Callable[[ParsedInput], GenericOutput]:
        pass

    @abstractmethod
    def parse_input(self, input_sample: DiarizationSample) -> ParsedInput:
        pass

    @abstractmethod
    def parse_output(self, output: GenericOutput) -> PipelineOutput:
        pass

    def __call__(self, input_sample: DiarizationSample) -> PipelineOutput:
        parsed_input = self.parse_input(input_sample)
        start_time = time.perf_counter()
        output = self.pipeline(parsed_input)
        end_time = time.perf_counter()
        prediction_time = end_time - start_time
        parsed_output = self.parse_output(output)
        # If `prediction_time` is not set after parsing the output, set it as the time taken to perform the diarization call
        if parsed_output.prediction_time is None:
            parsed_output.prediction_time = prediction_time
        return parsed_output
