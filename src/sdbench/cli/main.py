"""Main CLI entry point for sdbench-cli."""

import typer


app = typer.Typer(
    name="sdbench-cli",
    help="Benchmark suite for speaker diarization",
    add_completion=False,
)


@app.command()
def evaluate(
    config: str | None = typer.Option(None, "--config", "-c", help="Path to configuration file"),
    output_dir: str | None = typer.Option(None, "--output-dir", "-o", help="Output directory for results"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Run evaluation benchmarks."""
    # TODO: Implement evaluation logic
    typer.echo("Evaluation command - implementation pending")


@app.command()
def inference(
    model: str | None = typer.Option(None, "--model", "-m", help="Model to use for inference"),
    input_file: str | None = typer.Option(None, "--input", "-i", help="Input audio file"),
    output_file: str | None = typer.Option(None, "--output", "-o", help="Output file path"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Run inference on audio files."""
    # TODO: Implement inference logic
    typer.echo("Inference command - implementation pending")


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
