# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Shared utilities for pipeline implementations."""

import os
import subprocess
from pathlib import Path
from typing import Tuple

from argmaxtools.utils import _maybe_git_clone, get_logger


logger = get_logger(__name__)


class CLIError(Exception):
    """Raised when there's an error with CLI operations."""

    pass


def clone_and_build_swift_cli(
    out_dir: str | Path,
    repo_url: str,
    product_name: str,
    commit_hash: str | None = None,
) -> Tuple[str, str]:
    """
    Clone a repository and build its Swift CLI.

    Args:
        out_dir: Directory to clone the repo into
        repo_url: URL of the repository to clone
        product_name: Name of the Swift product to build
        commit_hash: Optional commit hash to checkout

    Returns:
        Tuple of (cli_path, commit_hash)

    Raises:
        CLIError: If build process fails
        ValueError: If repository URL is not set
    """
    os.makedirs(out_dir, exist_ok=True)
    if not repo_url:
        raise ValueError("Repository URL is not set")

    logger.info(f"Cloning repo {repo_url} into {out_dir}")
    repo_name = repo_url.split("/")[-1]
    repo_owner = repo_url.split("/")[-2]
    wkpro_ro = os.getenv("WKPRO_RO")
    hub_url = f"{wkpro_ro}@github.com" if wkpro_ro else "github.com"

    repo_dir, commit_hash = _maybe_git_clone(
        out_dir=out_dir,
        hub_url=hub_url,
        repo_name=repo_name,
        repo_owner=repo_owner,
        commit_hash=commit_hash,
    )
    logger.info(f"{repo_name} -> Commit hash: {commit_hash}")

    try:
        _init_submodules(repo_dir)
        build_dir = _build_cli(repo_dir, product_name)
        cli_path = os.path.join(build_dir, product_name)
        return cli_path, commit_hash
    except subprocess.CalledProcessError as e:
        logger.error(f"Build failed with return code {e.returncode}")
        logger.error(f"Build stdout:\n{e.stdout}")
        logger.error(f"Build stderr:\n{e.stderr}")
        raise CLIError(f"Failed to build CLI: Exit code {e.returncode}\nStdout: {e.stdout}\nStderr: {e.stderr}")


def _init_submodules(repo_dir: str) -> None:
    """Initialize and update git submodules."""
    logger.info("Initializing git submodules")
    subprocess.run(["git", "submodule", "init"], cwd=repo_dir)
    subprocess.run(["git", "submodule", "update", "--remote", "--merge"], cwd=repo_dir)


def _build_cli(repo_dir: str, product_name: str) -> str:
    """Build the CLI and return the build directory path."""
    logger.info(f"Building {product_name} CLI...")
    subprocess.run(
        f"swift build -c release --product {product_name}",
        cwd=repo_dir,
        shell=True,
        check=True,
    )
    logger.info(f"Successfully built {product_name} CLI!")

    result = subprocess.run(
        f"swift build -c release --product {product_name} --show-bin-path",
        cwd=repo_dir,
        stdout=subprocess.PIPE,
        shell=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()
