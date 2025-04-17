# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from typing import Callable

import torch
from argmaxtools.utils import get_fastest_device
from pyannote.audio import Pipeline as PyannotePipeline
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio.sample import Annotation
from pydantic import model_validator

from ....dataset import DiarizationSample
from ....pipeline_prediction import DiarizationAnnotation
from ...base import Pipeline, PipelineType, register_pipeline
from ..common import DiarizationOutput, DiarizationPipelineConfig
from .oracle_diarizer import OracleSpeakerDiarization

__all__ = ["PyAnnotePipeline", "PyAnnotePipelineConfig"]


class PyAnnotePipelineConfig(DiarizationPipelineConfig):
    device: str | None = None
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    use_oracle_clustering: bool | None = None
    use_oracle_segmentation: bool | None = None
    use_float16: bool = False

    @model_validator(mode="after")
    def resolve_device(self) -> "PyAnnotePipelineConfig":
        self.device = get_fastest_device()
        return self


@register_pipeline
class PyAnnotePipeline(Pipeline):
    _config_class = PyAnnotePipelineConfig
    pipeline_type = PipelineType.DIARIZATION

    def build_pipeline(
        self,
    ) -> Callable[[dict[str, torch.FloatTensor | int]], DiarizationAnnotation]:
        pipeline = PyannotePipeline.from_pretrained("pyannote/speaker-diarization-3.1")

        if self.config.use_oracle_clustering or self.config.use_oracle_segmentation:
            clustering = (
                "OracleClustering"
                if self.config.use_oracle_clustering
                else "AgglomerativeClustering"
            )
            pipeline = self._build_oracle_pipeline(pipeline, clustering)

        pipeline.to(torch.device(self.config.device))

        def call_pipeline(
            inputs: dict[str, torch.FloatTensor | int]
        ) -> DiarizationAnnotation:
            with torch.autocast(
                device_type=self.config.device,
                enabled=True,
                dtype=torch.float16 if self.config.use_float16 else torch.float32,
            ):
                annot: Annotation = pipeline(
                    inputs,
                    num_speakers=self.config.num_speakers,
                    min_speakers=self.config.min_speakers,
                    max_speakers=self.config.max_speakers,
                )
            return DiarizationAnnotation.from_pyannote_annotation(annot)

        return call_pipeline

    def _build_oracle_pipeline(self, pipeline: Pipeline, clustering: str) -> Pipeline:
        """Build oracle pipeline with specified clustering method."""
        if self.config.use_oracle_segmentation:
            _pipeline = OracleSpeakerDiarization(
                clustering=clustering,
                segmentation=pipeline.segmentation_model,
                embedding=pipeline.embedding,
                segmentation_step=pipeline.segmentation_step,
                embedding_batch_size=pipeline.embedding_batch_size,
                embedding_exclude_overlap=pipeline.embedding_exclude_overlap,
                der_variant=pipeline.der_variant,
            )
        else:
            _pipeline = SpeakerDiarization(
                clustering=clustering,
                segmentation=pipeline.segmentation_model,
                embedding=pipeline.embedding,
                segmentation_step=pipeline.segmentation_step,
                embedding_batch_size=pipeline.embedding_batch_size,
                embedding_exclude_overlap=pipeline.embedding_exclude_overlap,
                der_variant=pipeline.der_variant,
            )

        # Freeze and Instantiate
        params = {
            k: v
            for k, v in pipeline.parameters(instantiated=True).items()
            if k != "clustering" or not self.config.use_oracle_clustering
        }
        _pipeline.freeze(params)
        _pipeline.instantiate(params)
        return _pipeline

    def parse_input(
        self, input_sample: DiarizationSample
    ) -> dict[str, torch.FloatTensor | int]:
        waveform = torch.from_numpy(input_sample.waveform).float().unsqueeze(0)
        waveform = waveform.to(self.config.device)
        parsed_input = dict(
            waveform=waveform,
            sample_rate=input_sample.sample_rate,
        )
        if self.config.use_oracle_clustering or self.config.use_oracle_segmentation:
            parsed_input["annotation"] = input_sample.annotation
        return parsed_input

    def parse_output(self, output: DiarizationAnnotation) -> DiarizationOutput:
        return DiarizationOutput(prediction=output)
