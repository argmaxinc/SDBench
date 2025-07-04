# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.


import numpy as np
from pydantic import Field

from ...pipeline_prediction import DiarizationAnnotation
from ..base import PipelineConfig, PipelineOutput


class DiarizationPipelineConfig(PipelineConfig):
    max_speakers: int | None = Field(
        default=None,
        description="The maximum number of speakers to diarize",
    )
    min_num_speakers: int | None = Field(
        default=None,
        description="The minimum number of speakers to diarize",
    )
    use_exact_num_speakers: bool = Field(
        default=False,
        description="Whether to use the exact number of speakers if an annotation is provided",
    )


class DiarizationOutput(PipelineOutput[DiarizationAnnotation]):
    embeddings: np.ndarray | None = Field(
        None,
        description="The embeddings used for clustering `(num_embeddings, embedding_dim)` in case pipeline provides it",
    )
    cluster_labels: np.ndarray | None = Field(
        None,
        description="The cluster labels assigned to each embedding `(num_embeddings, )` in case pipeline provides it",
    )
    centroids: np.ndarray | None = Field(
        None,
        description="The centroids for each cluster with shape `(num_clusters, embedding_dim)` in case pipeline provides it",
    )

    class Config:
        arbitrary_types_allowed = True
