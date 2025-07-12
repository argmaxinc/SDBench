# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Summary command for sdbench-cli."""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sdbench.dataset import DatasetRegistry
from sdbench.metric import MetricRegistry
from sdbench.pipeline import PipelineRegistry
from sdbench.types import PipelineType


console = Console()


def create_pipeline_table() -> Table:
    """Create a table showing available pipelines."""
    table = Table(title="Available Pipelines", show_header=True, header_style="bold magenta")
    table.add_column("Class Name", style="cyan", no_wrap=True)
    table.add_column("Type", style="green")
    table.add_column("Alias", style="yellow")
    table.add_column("Alias Description", style="white")

    # Get all pipeline names (both class names and aliases)
    pipeline_names = PipelineRegistry.list_pipelines()

    # Group by class name to show aliases together
    class_to_aliases = {}
    for name in pipeline_names:
        try:
            if PipelineRegistry.is_alias(name):
                # Get the class name for this alias
                alias_info = PipelineRegistry.get_alias_info(name)
                class_name = alias_info.pipeline_class.__name__
                if class_name not in class_to_aliases:
                    class_to_aliases[class_name] = []
                class_to_aliases[class_name].append((name, alias_info.description or "No description available"))
            else:
                # This is a class name, add it with no aliases
                if name not in class_to_aliases:
                    class_to_aliases[name] = []
        except Exception:
            # Skip pipelines that can't be processed
            continue

    # Add rows for each class
    for class_name, aliases in class_to_aliases.items():
        try:
            pipeline_type = PipelineRegistry.get_pipeline_type(class_name)
            pipeline_type_str = pipeline_type.name if pipeline_type else "Unknown"

            if aliases:
                # Show aliases for this class
                alias_names = [alias[0] for alias in aliases]
                alias_descriptions = [alias[1] for alias in aliases]
                alias_str = ", ".join(alias_names)
                description_str = "; ".join(alias_descriptions)
            else:
                # No aliases for this class
                alias_str = "-"
                description_str = "-"

            table.add_row(class_name, pipeline_type_str, alias_str, description_str)
        except Exception:
            # Skip pipelines that can't be processed
            continue

    return table


def create_dataset_table() -> Table:
    """Create a table showing available datasets."""
    table = Table(title="Available Datasets", show_header=True, header_style="bold magenta")
    table.add_column("Alias", style="cyan", no_wrap=True)
    table.add_column("Dataset ID", style="green")
    table.add_column("Description", style="white")
    table.add_column("Pipeline Types", style="yellow")

    # Get all dataset aliases
    dataset_aliases = DatasetRegistry.list_aliases()

    for alias in dataset_aliases:
        try:
            alias_info = DatasetRegistry.get_alias_info(alias)
            dataset_id = alias_info.config.dataset_id
            description = alias_info.description or "No description available"

            # Get compatible pipeline types
            supported_types = alias_info.supported_pipeline_types
            pipeline_types_str = ", ".join([pt.name for pt in supported_types]) if supported_types else "All"

            table.add_row(alias, dataset_id, description, pipeline_types_str)
        except Exception:
            # Skip datasets that can't be processed
            continue

    return table


def create_metric_table() -> Table:
    """Create a table showing available metrics."""
    table = Table(title="Available Metrics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Pipeline Types", style="yellow")

    # Get metrics for each pipeline type
    all_metrics = set()

    for pipeline_type in PipelineType:
        try:
            metrics = MetricRegistry.get_available_metrics(pipeline_type)
            all_metrics.update(metrics)
        except Exception:
            continue

    # Create list of metric info for sorting
    metric_info_list = []
    for metric_option in all_metrics:
        try:
            metric_name = metric_option.value
            description = f"Available for {metric_option.value} metric"

            # Find which pipeline types support this metric
            supported_pipeline_types = []
            for pipeline_type in PipelineType:
                try:
                    metrics = MetricRegistry.get_available_metrics(pipeline_type)
                    if metric_option in metrics:
                        supported_pipeline_types.append(pipeline_type.name)
                except Exception:
                    continue

            # Sort the supported pipeline types for consistent ordering
            supported_pipeline_types.sort()
            pipeline_types_str = ", ".join(supported_pipeline_types) if supported_pipeline_types else "Unknown"

            # Store metric info for sorting
            metric_info_list.append(
                {
                    "metric_name": metric_name,
                    "description": description,
                    "pipeline_types_str": pipeline_types_str,
                    "supported_pipeline_types": supported_pipeline_types,
                }
            )
        except Exception:
            # Skip metrics that can't be processed
            continue

    # Sort by pipeline types first, then by metric name
    metric_info_list.sort(key=lambda x: (x["supported_pipeline_types"], x["metric_name"]))

    # Add sorted metrics to table
    for metric_info in metric_info_list:
        table.add_row(metric_info["metric_name"], metric_info["description"], metric_info["pipeline_types_str"])

    return table


def create_compatibility_table() -> Table:
    """Create a table showing pipeline-dataset compatibility."""
    table = Table(title="Pipeline-Dataset Compatibility", show_header=True, header_style="bold magenta")
    table.add_column("Pipeline Class", style="cyan", no_wrap=True)
    table.add_column("Compatible Datasets", style="green")
    table.add_column("Compatible Metrics", style="yellow")

    # Get all pipeline names and filter to only class names (not aliases)
    pipeline_names = PipelineRegistry.list_pipelines()
    dataset_aliases = DatasetRegistry.list_aliases()

    for pipeline_name in pipeline_names:
        # Skip aliases, only show class names
        if PipelineRegistry.is_alias(pipeline_name):
            continue

        try:
            pipeline_type = PipelineRegistry.get_pipeline_type(pipeline_name)

            # Find compatible datasets
            compatible_datasets = []
            for dataset_alias in dataset_aliases:
                try:
                    dataset_info = DatasetRegistry.get_alias_info(dataset_alias)
                    if pipeline_type in dataset_info.supported_pipeline_types:
                        compatible_datasets.append(dataset_alias)
                except Exception:
                    continue

            # Find compatible metrics
            compatible_metrics = []
            try:
                metrics = MetricRegistry.get_available_metrics(pipeline_type)
                compatible_metrics = [metric.value for metric in metrics]
            except Exception:
                pass

            datasets_str = ", ".join(compatible_datasets[:3])  # Show first 3
            if len(compatible_datasets) > 3:
                datasets_str += f" (+{len(compatible_datasets) - 3} more)"

            metrics_str = ", ".join(compatible_metrics[:3])  # Show first 3
            if len(compatible_metrics) > 3:
                metrics_str += f" (+{len(compatible_metrics) - 3} more)"

            table.add_row(pipeline_name, datasets_str, metrics_str)
        except Exception:
            # Skip pipelines that can't be processed
            continue

    return table


def summary(
    pipelines: bool = typer.Option(True, "--pipelines", "-p", help="Show available pipelines"),
    datasets: bool = typer.Option(True, "--datasets", "-d", help="Show available datasets"),
    metrics: bool = typer.Option(True, "--metrics", "-m", help="Show available metrics"),
    compatibility: bool = typer.Option(True, "--compatibility", "-c", help="Show compatibility matrix"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information"),
) -> None:
    """Show a summary of available pipelines, datasets, and metrics.

    This command provides an overview of what's available in SDBench,
    including compatibility information between pipelines, datasets, and metrics.

    Examples:

        # Show everything (default)
        sdbench-cli summary

        # Show only pipelines
        sdbench-cli summary --pipelines --no-datasets --no-metrics --no-compatibility

        # Show only compatibility matrix
        sdbench-cli summary --no-pipelines --no-datasets --no-metrics --compatibility
    """
    console.print(Panel.fit(Text("SDBench Summary", style="bold blue"), border_style="blue"))

    if pipelines:
        console.print()
        console.print(create_pipeline_table())

    if datasets:
        console.print()
        console.print(create_dataset_table())

    if metrics:
        console.print()
        console.print(create_metric_table())

    if compatibility:
        console.print()
        console.print(create_compatibility_table())

    if verbose:
        console.print()
        console.print(
            Panel(
                Text(
                    "💡 Tip: Use --help with any command to see detailed usage information\n"
                    "📖 Examples:\n"
                    "  sdbench-cli evaluate --help\n"
                    "  sdbench-cli inference --help\n"
                    "  sdbench-cli summary --help",
                    style="dim",
                ),
                title="💡 Help & Tips",
                border_style="green",
            )
        )
