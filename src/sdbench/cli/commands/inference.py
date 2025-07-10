# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Inference command for sdbench-cli."""

import typer


def inference(
    model: str | None = typer.Option(None, "--model", "-m", help="Model to use for inference"),
    input_file: str | None = typer.Option(None, "--input", "-i", help="Input audio file"),
    output_file: str | None = typer.Option(None, "--output", "-o", help="Output file path"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Run inference on audio files."""
    # TODO: Implement inference logic
    typer.echo("Inference command - implementation pending")
