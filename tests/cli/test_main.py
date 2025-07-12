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


def test_evaluation_command_help(runner):
    """Test that the evaluation command shows help."""
    result = runner.invoke(app, ["evaluation", "--help"])
    assert result.exit_code == 0
    assert "Run evaluation benchmarks" in result.output


def test_inference_command_help(runner):
    """Test that the inference command shows help."""
    result = runner.invoke(app, ["inference", "--help"])
    assert result.exit_code == 0
    assert "Run inference on audio files" in result.output


def test_evaluation_command_placeholder(runner):
    """Test that the evaluation command returns placeholder message."""
    result = runner.invoke(app, ["evaluation"])
    assert result.exit_code == 0
    assert "Evaluation command - implementation pending" in result.output


def test_inference_command_placeholder(runner):
    """Test that the inference command returns placeholder message."""
    result = runner.invoke(app, ["inference"])
    assert result.exit_code == 0
    assert "Inference command - implementation pending" in result.output
