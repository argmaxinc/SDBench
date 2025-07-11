# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Evaluate command for sdbench-cli."""

import typer

from sdbench.dataset import DatasetRegistry
from sdbench.metric import MetricOptions
from sdbench.pipeline import PipelineRegistry
from sdbench.runner import BenchmarkConfig, BenchmarkRunner, WandbConfig

from ..command_utils import (
    get_datasets_help_text,
    get_metrics_help_text,
    get_pipelines_help_text,
    validate_dataset_name,
    validate_pipeline_dataset_compatibility,
    validate_pipeline_metrics_compatibility,
    validate_pipeline_name,
)


def evaluate(
    pipeline_name: str = typer.Option(
        ...,
        "--pipeline",
        "-p",
        help=f"The name of the registered pipeline to use for evaluation\n\n{get_pipelines_help_text()}",
        callback=validate_pipeline_name,
    ),
    dataset_name: str = typer.Option(
        ...,
        "--dataset",
        "-d",
        help=f"The alias of the registered dataset to use for evaluation\n\n{get_datasets_help_text()}",
        callback=validate_dataset_name,
    ),
    metrics: list[MetricOptions] = typer.Option(
        ...,
        "--metrics",
        "-m",
        help=f"The metrics to use for evaluation\n\n{get_metrics_help_text()}",
    ),
    output_dir: str = typer.Option(".", "--output-dir", "-o", help="Output directory for results"),
    ######## WandB arguments ########
    use_wandb: bool = typer.Option(False, "--use-wandb", "-w", help="Use W&B for evaluation"),
    wandb_project: str = typer.Option(
        "sdbench-eval", "--wandb-project", "-wp", help="W&B project to use for evaluation"
    ),
    wandb_run_name: str | None = typer.Option(
        None, "--wandb-run-name", "-wr", help="W&B run name to use for evaluation"
    ),
    wandb_tags: list[str] | None = typer.Option(None, "--wandb-tags", "-wt", help="W&B tags to use for evaluation"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Run evaluation benchmarks.

    This command evaluates a pipeline on a dataset using specified metrics.

    Examples:
        # Evaluate pyannote pipeline on voxconverse dataset with DER and JER metrics
        sdbench-cli evaluate run --pipeline PyAnnotePipeline --dataset voxconverse --metrics der jer

        # Evaluate with WandB logging
        sdbench-cli evaluate run --pipeline PyAnnotePipeline --dataset voxconverse --metrics der jer --use-wandb --wandb-project my-project
    """
    # Validate cross-parameter compatibility
    validate_pipeline_dataset_compatibility(pipeline_name, dataset_name)
    validate_pipeline_metrics_compatibility(pipeline_name, metrics)

    if verbose:
        dataset_info = DatasetRegistry.get_alias_info(dataset_name)
        typer.echo(f"✅ Pipeline: {pipeline_name}")
        typer.echo(f"✅ Dataset: {dataset_name} ({dataset_info.config.dataset_id})")
        typer.echo(f"✅ Metrics: {[m.value for m in metrics]}")
        typer.echo(f"✅ Output directory: {output_dir}")
        typer.echo(f"✅ WandB: {'enabled' if use_wandb else 'disabled'}")

    ######### Build Pipeline #########
    pipeline = PipelineRegistry.create_pipeline(pipeline_name)

    ######### Build Benchmark Config #########
    dataset_config = DatasetRegistry.get_alias_config(dataset_name)

    wandb_config = WandbConfig(
        project_name=wandb_project,
        run_name=wandb_run_name,
        tags=wandb_tags,
        is_active=use_wandb,
    )

    benchmark_config = BenchmarkConfig(
        wandb_config=wandb_config, datasets={dataset_name: dataset_config}, metrics={metric: {} for metric in metrics}
    )

    # Create runner
    benchmark_runner = BenchmarkRunner(config=benchmark_config, pipelines=[pipeline])

    # TODO: Run the benchmark
    typer.echo("🚀 Starting evaluation...")
    benchmark_result = benchmark_runner.run()
    print(benchmark_result)
    typer.echo("✅ Evaluation completed successfully!")
