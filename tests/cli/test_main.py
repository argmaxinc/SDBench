"""Tests for the CLI main module."""

import pytest
from typer.testing import CliRunner

from sdbench.cli.main import app


@pytest.fixture
def runner():
    """Create a CLI runner for testing."""
    return CliRunner()


def test_cli_help(runner):
    """Test that the CLI shows help."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Benchmark suite for speaker diarization" in result.output


def test_evaluate_command_help(runner):
    """Test that the evaluate command shows help."""
    result = runner.invoke(app, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "evaluate" in result.output


def test_inference_command_help(runner):
    """Test that the inference command shows help."""
    result = runner.invoke(app, ["inference", "--help"])
    assert result.exit_code == 0
    assert "inference" in result.output


def test_summary_command_help(runner):
    """Test that the summary command shows help."""
    result = runner.invoke(app, ["summary", "--help"])
    assert result.exit_code == 0
    assert "summary" in result.output


def test_evaluate_command_requires_pipeline(runner):
    """Test that the evaluate command requires pipeline parameter."""
    result = runner.invoke(app, ["evaluate"])
    assert result.exit_code != 0
    assert "Missing option" in result.output or "Error" in result.output


def test_inference_command_requires_pipeline_and_audio(runner):
    """Test that the inference command requires pipeline and audio parameters."""
    result = runner.invoke(app, ["inference"])
    assert result.exit_code != 0
    assert "Missing option" in result.output or "Error" in result.output


def test_summary_command_default(runner):
    """Test that the summary command runs successfully."""
    result = runner.invoke(app, ["summary"])
    assert result.exit_code == 0
    assert "SDBench Summary" in result.output


def test_summary_command_pipelines_only(runner):
    """Test that the summary command shows only pipelines when others are disabled."""
    result = runner.invoke(app, ["summary", "--disable-datasets", "--disable-metrics", "--disable-compatibility"])
    assert result.exit_code == 0
    assert "Available Pipelines" in result.output
    assert "Available Datasets" not in result.output
    assert "Available Metrics" not in result.output
    assert "Pipeline-Dataset Compatibility" not in result.output
