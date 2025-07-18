# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Dataset alias registrations for the CLI."""

from ..types import PipelineType
from .dataset_base import DatasetConfig
from .dataset_registry import DatasetRegistry


def register_dataset_aliases() -> None:
    """Register all dataset aliases with their configurations."""

    ########## DIARIZATION ##########
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

    ########## TRANSCRIPTION ##########

    DatasetRegistry.register_alias(
        "librispeech",
        DatasetConfig(dataset_id="argmaxinc/librispeech-openbench", split="test", subset="full"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="LibriSpeech dataset for transcription evaluation",
    )

    DatasetRegistry.register_alias(
        "librispeech-200",
        DatasetConfig(dataset_id="argmaxinc/librispeech-openbench", split="test", subset="200"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="LibriSpeech dataset for transcription evaluation with only 200 files. Commonly used for debugging or get an estimate of the performance of the model.",
    )

    DatasetRegistry.register_alias(
        "earnings22",
        DatasetConfig(dataset_id="argmaxinc/earnings22-openbench", split="test", subset="full"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Earnings call dataset for transcription evaluation.",
    )

    DatasetRegistry.register_alias(
        "earnings22-12hours",
        DatasetConfig(dataset_id="argmaxinc/earnings22-openbench", split="test", subset="12hours"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Earnings call dataset for transcription evaluation with only 12 hours of audio.",
    )

    DatasetRegistry.register_alias(
        "earnings22-3hours",
        DatasetConfig(dataset_id="argmaxinc/earnings22-openbench", split="test", subset="3hours"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Earnings call dataset for transcription evaluation with only 3 hours of audio.",
    )

    DatasetRegistry.register_alias(
        "common-voice",
        DatasetConfig(
            dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="full"
        ),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains all languages",
    )

    DatasetRegistry.register_alias(
        "common-voice",
        DatasetConfig(
            dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="full"
        ),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains all languages",
    )

    # Common Voice specific languages subsets

    # English
    DatasetRegistry.register_alias(
        "common-voice-en",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="en"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only english",
    )

    # Spanish
    DatasetRegistry.register_alias(
        "common-voice-es",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="es"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only spanish",
    )

    # German
    DatasetRegistry.register_alias(
        "common-voice-de",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="de"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only german",
    )

    # French
    DatasetRegistry.register_alias(
        "common-voice-fr",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="fr"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only french",
    )

    # Portuguese
    DatasetRegistry.register_alias(
        "common-voice-pt",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="pt"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only portuguese",
    )

    # Japanese
    DatasetRegistry.register_alias(
        "common-voice-ja",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="ja"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only japanese",
    )

    # Italian
    DatasetRegistry.register_alias(
        "common-voice-it",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="it"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only italian",
    )

    # Chinese
    DatasetRegistry.register_alias(
        "common-voice-zh",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="zh"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only chinese",
    )

    # Dutch
    DatasetRegistry.register_alias(
        "common-voice-nl",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="nl"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only dutch",
    )

    # Polish
    DatasetRegistry.register_alias(
        "common-voice-pl",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="pl"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only polish",
    )

    # Indonesian
    DatasetRegistry.register_alias(
        "common-voice-id",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="id"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only indonesian",
    )

    # Galician
    DatasetRegistry.register_alias(
        "common-voice-gl",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="gl"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only galician",
    )

    # Romanian
    DatasetRegistry.register_alias(
        "common-voice-ro",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="ro"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only romanian",
    )

    # Czech
    DatasetRegistry.register_alias(
        "common-voice-cs",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="cs"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only czech",
    )

    # Swedish
    DatasetRegistry.register_alias(
        "common-voice-sv",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="sv"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only swedish",
    )

    # Hungarian
    DatasetRegistry.register_alias(
        "common-voice-hu",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="hu"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only hungarian",
    )

    # Greek
    DatasetRegistry.register_alias(
        "common-voice-el",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="el"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only greek",
    )

    # Finnish
    DatasetRegistry.register_alias(
        "common-voice-fi",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="fi"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only finnish",
    )

    # Vietnamese
    DatasetRegistry.register_alias(
        "common-voice-vi",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="vi"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only vietnamese",
    )

    # Danish
    DatasetRegistry.register_alias(
        "common-voice-da",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="da"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only danish",
    )

    # Catalan
    DatasetRegistry.register_alias(
        "common-voice-ca",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="ca"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only catalan",
    )

    ########## STREAMING TRANSCRIPTION ##########

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


register_dataset_aliases()
