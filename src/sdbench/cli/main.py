# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Main CLI entry point for sdbench-cli."""

import typer

from .commands import evaluate, inference


app = typer.Typer(
    name="sdbench-cli",
    help="Benchmark suite for speaker diarization",
    add_completion=False,
)

# Add commands to the app
app.command()(evaluate)
app.command()(inference)


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
