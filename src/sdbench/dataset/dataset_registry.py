# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from typing import ClassVar

from ..pipeline.utils import PipelineType
from .dataset_base import BaseDataset, DatasetConfig
from .dataset_diarization import DiarizationDataset
from .dataset_orchestration import OrchestrationDataset
from .dataset_streaming_transcription import StreamingDataset
from .dataset_transcription import TranscriptionDataset


class DatasetRegistry:
    """Registry for datasets by pipeline type."""

    _datasets: ClassVar[dict[PipelineType, type[BaseDataset]]] = {}

    @classmethod
    def register(cls, pipeline_type: PipelineType, dataset_class: type[BaseDataset]) -> None:
        """Register a dataset class for a specific pipeline type."""
        cls._datasets[pipeline_type] = dataset_class

    @classmethod
    def get_dataset_for_pipeline(cls, pipeline_type: PipelineType, config: DatasetConfig) -> BaseDataset:
        """Get a dataset instance for a specific pipeline type."""
        if pipeline_type not in cls._datasets:
            raise KeyError(f"No dataset registered for pipeline type: {pipeline_type}")

        dataset_class = cls._datasets[pipeline_type]
        return dataset_class.from_config(config)

    @classmethod
    def get_expected_columns(cls, pipeline_type: PipelineType) -> list[str]:
        """Get the expected columns for a specific pipeline type."""
        if pipeline_type not in cls._datasets:
            raise KeyError(f"No dataset registered for pipeline type: {pipeline_type}")

        dataset_class = cls._datasets[pipeline_type]
        return dataset_class._expected_columns


# Register all datasets
DatasetRegistry.register(PipelineType.DIARIZATION, DiarizationDataset)
DatasetRegistry.register(PipelineType.ORCHESTRATION, OrchestrationDataset)
DatasetRegistry.register(PipelineType.STREAMING_TRANSCRIPTION, StreamingDataset)
DatasetRegistry.register(PipelineType.TRANSCRIPTION, TranscriptionDataset)
