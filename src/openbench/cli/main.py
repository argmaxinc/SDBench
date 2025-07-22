# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Main CLI entry point for openbench-cli."""

import typer

from openbench.cli.commands import evaluate, inference, summary


app = typer.Typer(
    name="openbench-cli",
    help="OpenBench CLI for benchmarking, currently supports transcription, diarization, diarized-transcripts (aka orchestration) and realtime-transcription",
    add_completion=False,
)

# Add commands to the app
app.command()(evaluate)
app.command()(inference)
app.command()(summary)


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
