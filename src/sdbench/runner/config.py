# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from typing import Any

from pydantic import BaseModel, Field

from ..dataset import DiarizationDatasetConfig
from ..metric import MetricOptions


class WandbConfig(BaseModel):
    project_name: str = Field(..., description="The name of the project for wandb")
    run_name: str | None = Field(None, description="The name of the run for wandb")
    is_active: bool = Field(True, description="Whether to log to wandb")
    tags: list[str] | None = Field(None, description="Tags for the run")

    @property
    def wandb_mode(self) -> str:
        return "disabled" if not self.is_active else "online"


class BenchmarkConfig(BaseModel):
    wandb_config: WandbConfig = Field(..., description="The wandb config")
    # a dictionary mapping tasks types to metric options and their initialization kwargs
    metrics: dict[MetricOptions, dict[str, Any]] = Field(
        ..., description="The metrics that will be used for each task"
    )
    datasets: dict[str, DiarizationDatasetConfig] = Field(
        ..., description="Datasets to evaluate"
    )

    class Config:
        arbitrary_types_allowed = True

    def get_wandb_config_to_log(self) -> dict[str, Any]:
        wandb_config: dict[str, Any] = self.model_dump()
        # Convert `metrics` that use enums to their respective values
        wandb_config["metrics"] = {
            metric.value: kwargs for metric, kwargs in wandb_config["metrics"].items()
        }
        return wandb_config
