# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from pathlib import Path
from typing import Optional, Text, Union

from pyannote.audio import Model
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio.pipelines.utils import PipelineModel, get_model

from .oracle_inference import OracleSegmenterInference


class OracleSpeakerDiarization(SpeakerDiarization):
    """Oracle speaker diarization pipeline that can use ground truth segmentation."""

    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/segmentation@2022.07",
        segmentation_step: float = 0.1,
        embedding: PipelineModel = "speechbrain/spkrec-ecapa-voxceleb@5c0be3875fda05e81f3c004ed8c7c06be308de1e",
        embedding_exclude_overlap: bool = False,
        clustering: str = "AgglomerativeClustering",
        embedding_batch_size: int = 1,
        segmentation_batch_size: int = 1,
        der_variant: Optional[dict] = None,
        use_auth_token: Union[Text, None] = None,
    ):
        super().__init__(
            segmentation,
            segmentation_step,
            embedding,
            embedding_exclude_overlap,
            clustering,
            embedding_batch_size,
            segmentation_batch_size,
            der_variant,
            use_auth_token,
        )

        model: Model = get_model(segmentation, use_auth_token=use_auth_token)
        self.segmentation_step = segmentation_step
        segmentation_duration = model.specifications.duration

        self._segmentation = OracleSegmenterInference(
            model,
            duration=segmentation_duration,
            step=self.segmentation_step * segmentation_duration,
            skip_aggregation=True,
            batch_size=segmentation_batch_size,
        )
