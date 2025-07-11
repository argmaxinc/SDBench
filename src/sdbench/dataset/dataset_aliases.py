# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Dataset alias registrations for the CLI."""

from ..types import PipelineType
from .dataset_base import DatasetConfig
from .dataset_registry import DatasetRegistry


def register_dataset_aliases() -> None:
    """Register all dataset aliases with their configurations."""

    # Diarization and Orchestration datasets
    DatasetRegistry.register_alias(
        "voxconverse",
        DatasetConfig(dataset_id="diarizers-community/voxconverse", split="test"),
        supported_pipeline_types={
            PipelineType.DIARIZATION,
        },
        description="Speaker diarization dataset with ground truth annotations",
    )

    DatasetRegistry.register_alias(
        "callhome_hf",
        DatasetConfig(dataset_id="talkbank/callhome", split="data", subset="eng", num_samples=1),
        supported_pipeline_types={
            PipelineType.DIARIZATION,
        },
        description="Talkbank provided Callhome dataset english subset with 1 sample for debugging purposes. Note that this dataset is gated on HF Hub.",
    )

    DatasetRegistry.register_alias(
        "msdwild",
        DatasetConfig(dataset_id="argmaxinc/msdwild", split="test"),
        supported_pipeline_types={
            PipelineType.DIARIZATION,
        },
        description="Multi-speaker dataset with diverse scenarios",
    )

    DatasetRegistry.register_alias(
        "earnings21",
        DatasetConfig(dataset_id="argmaxinc/earnings21", split="test"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
            PipelineType.ORCHESTRATION,
            PipelineType.DIARIZATION,
        },
        description="Earnings call dataset with transcription ground truth",
    )

    DatasetRegistry.register_alias(
        "ami-ihm",
        DatasetConfig(dataset_id="diarizers-community/ami", split="test", subset="ihm"),
        supported_pipeline_types={
            PipelineType.DIARIZATION,
        },
        description="AMI meeting dataset with IHM microphone setup",
    )

    DatasetRegistry.register_alias(
        "ami-sdm",
        DatasetConfig(dataset_id="diarizers-community/ami", split="test", subset="sdm"),
        supported_pipeline_types={
            PipelineType.DIARIZATION,
        },
        description="AMI meeting dataset with SDM microphone setup",
    )

    DatasetRegistry.register_alias(
        "american-life-podcast",
        DatasetConfig(dataset_id="argmaxinc/american-life", split="test"),
        supported_pipeline_types={
            PipelineType.DIARIZATION,
        },
        description="This American Life podcast dataset",
    )

    DatasetRegistry.register_alias(
        "ava-avd",
        DatasetConfig(dataset_id="argmaxinc/ava-avd", split="test"),
        supported_pipeline_types={
            PipelineType.DIARIZATION,
        },
        description="AVA Audio-Visual Diarization dataset",
    )

    DatasetRegistry.register_alias(
        "timit",
        DatasetConfig(dataset_id="kylelovesllms/timit_asr", split="test", num_samples=300),
        supported_pipeline_types={PipelineType.STREAMING_TRANSCRIPTION},
        description="TIMIT dataset for streaming transcription evaluation",
    )

    DatasetRegistry.register_alias(
        "timit-debug",
        DatasetConfig(dataset_id="kylelovesllms/timit_asr", split="test", num_samples=5),
        supported_pipeline_types={PipelineType.STREAMING_TRANSCRIPTION},
        description="TIMIT dataset for streaming transcription evaluation for debugging purposes only",
    )

    DatasetRegistry.register_alias(
        "timit-stitched",
        DatasetConfig(dataset_id="argmaxinc/timit_stitched", split="test"),
        supported_pipeline_types={PipelineType.STREAMING_TRANSCRIPTION},
        description="TIMIT stitched dataset for streaming transcription evaluation",
    )

    DatasetRegistry.register_alias(
        "icsi",
        DatasetConfig(dataset_id="argmaxinc/icsi", split="test"),
        supported_pipeline_types={
            PipelineType.DIARIZATION,
        },
        description="ICSI meeting corpus dataset",
    )

    DatasetRegistry.register_alias(
        "aishell-4",
        DatasetConfig(dataset_id="argmaxinc/aishell-4", split="test"),
        supported_pipeline_types={
            PipelineType.DIARIZATION,
        },
        description="AISHELL-4 dataset for speaker diarization",
    )

    DatasetRegistry.register_alias(
        "ali-meetings",
        DatasetConfig(dataset_id="argmaxinc/ali-meetings", split="test"),
        supported_pipeline_types={
            PipelineType.DIARIZATION,
        },
        description="AliMeetings dataset for speaker diarization",
    )


register_dataset_aliases()
