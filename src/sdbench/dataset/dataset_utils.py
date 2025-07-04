# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from datasets import Dataset as HfDataset


def validate_hf_dataset_schema(ds: HfDataset, expected_columns: list[str]) -> None:
    """Validate that the dataset has the expected columns."""
    for col in expected_columns:
        if col not in ds.column_names:
            raise ValueError(f"Dataset is missing expected column: {col}") 