# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import wandb
from argmaxtools.utils import get_logger
from huggingface_hub import snapshot_download, upload_folder
from tqdm import tqdm

logger = get_logger(__name__)


def get_wandb_table_as_df(
    files: list[wandb.apis.public.files.File], filename: str, run_name: str
) -> pd.DataFrame:
    # Get the file matching the filename
    f = [f for f in files if filename in f.name]
    if len(f) == 0:
        raise ValueError(f"File {filename} not found in run {run_name}")
    if len(f) > 1:
        raise ValueError(
            f"Multiple files found with name {filename} in run {run_name} - {f}"
        )

    f = f[0]

    if not f.name.endswith("table.json"):
        raise ValueError(
            f"File {filename} is not a table file in run {run_name} - {f.name}"
        )

    # Downloads a wandb table file
    path = f.download(replace=True)
    path.seek(0)
    content = path.read()
    json_content = json.loads(content)
    columns = json_content["columns"]
    data = json_content["data"]
    return pd.DataFrame(data, columns=columns)


def preprocess_task_results_table(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # Preprocess the task results table and returns a dictionary with a dataframe for each metric
    unique_metrics = df["metric_name"].unique()
    dataframes = dict()
    logger.info(f"Processing {len(unique_metrics)} unique metrics")

    for metric in tqdm(unique_metrics, desc="Processing metrics"):
        df_subset = (
            df.query("metric_name == @metric")
            .dropna(how="all", axis=1)
            .reset_index(drop=True)
        )
        df_subset.columns = [
            col if "detailed" not in col else col.replace("detailed_", "")
            for col in df_subset.columns
        ]
        # Support legacy runs that have task_type column
        cols_to_drop = ["metric_name", "result"]
        if "task_type" in df_subset.columns:
            cols_to_drop = ["metric_name", "result", "task_type"]
        df_subset = df_subset.assign(**{metric: df_subset["result"]}).drop(
            columns=cols_to_drop
        )
        dataframes[metric] = df_subset
    return dataframes


def preprocess_diarization_prediction_table(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Preprocessing diarization prediction table")
    # This is to support legacy runs that have the embeddings_projection and prediction columns
    columns_to_drop = ["prediction", "embeddings_projection"]
    if all(col in df.columns for col in columns_to_drop):
        df = df.drop(columns=columns_to_drop)
    return df


def add_run_info(df: pd.DataFrame, run: wandb.apis.public.runs.Run) -> pd.DataFrame:
    logger.info(f"Adding run info for run {run.name}")
    pipeline_name = run.config["pipeline_name"]

    dataset_name = list(run.config["datasets"].keys())
    if len(dataset_name) > 1:
        raise ValueError(
            f"More than one dataset found in run {run.name} - {dataset_name}"
        )
    dataset_name = dataset_name[0]

    tags = run.tags
    experiment_tag = [tag for tag in tags if tag not in [dataset_name, pipeline_name]]
    # Add run info to the dataframe
    df["run_name"] = run.name
    df["run_id"] = run.id
    df["experiment_tag"] = experiment_tag[0]
    df["created_at"] = pd.Timestamp(run.created_at).tz_convert("UTC")
    return df


def save_or_append_to_parquet(df: pd.DataFrame, path: Path) -> None:
    logger.info(f"Saving/appending data to {path}")
    # If it doesn't exist, save it
    if not path.exists():
        df.to_parquet(path)
        logger.info(f"Created new parquet file at {path}")
    # If it does exist, load it, append it and save it
    else:
        df_existing = pd.read_parquet(path)
        df = pd.concat([df_existing, df], ignore_index=True)
        df.to_parquet(path)
        logger.info(f"Appended data to existing parquet file at {path}")


def save_processed_data(
    task_tables_merged: dict[str, pd.DataFrame],
    prediction_table_merged: pd.DataFrame,
    output_dir: str,
) -> None:
    logger.info("Saving processed data")
    diarization_prediction_table_path = (
        output_dir / "diarization_prediction_table.parquet"
    )
    save_or_append_to_parquet(
        prediction_table_merged, diarization_prediction_table_path
    )

    for metric, df in tqdm(task_tables_merged.items(), desc="Saving metric tables"):
        save_or_append_to_parquet(df, output_dir / f"per_sample_{metric}.parquet")


def download_preprocessed_data(
    repo_id: str, dir_to_download: str, output_dir: str
) -> str:
    # Downloads the previously processed data from a HF repo stored at `dir_to_download`
    # and saves it in the output directory
    logger.info(f"Downloading preprocessed data from {repo_id} to {output_dir}")
    path = snapshot_download(
        repo_id, local_dir=output_dir, allow_patterns=f"{dir_to_download}/*"
    )
    return path


def push_preprocessed_data(repo_id: str, dir_to_push: str, path_in_repo: str) -> None:
    # Pushes preprocessed data
    logger.info(f"Pushing preprocessed data to {repo_id} from {dir_to_push}")
    upload_folder(repo_id=repo_id, folder_path=dir_to_push, path_in_repo=path_in_repo)
    logger.info(f"Pushed preprocessed data to {repo_id} from {dir_to_push}")


def wandb_data_processor(
    project: str, entity: str = "speakerkit", output_dir: str = "wandb_data"
) -> None:
    logger.info(f"Starting data processing for project {project}")
    # Create the output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory at {output_dir}")

    # this .csv is just a single csv with the run-id records that have been processed
    # The name of the column is `run_id`
    processed_runs_path = output_dir / "processed_runs.csv"
    processed_runs = []
    if processed_runs_path.exists():
        processed_runs = pd.read_csv(processed_runs_path)["run_id"].tolist()
        logger.info(f"Found {len(processed_runs)} previously processed runs")

    api = wandb.Api()
    runs = api.runs(
        f"{entity}/{project}",
        filters={
            "state": "finished",
        },
    )
    runs = [run for run in runs if run.id not in processed_runs]
    if len(runs) == 0:
        logger.info("No new runs to process")
        return

    logger.info(f"Found {len(runs)} new runs to process")

    tasks_tables: dict[str, list[pd.DataFrame]] = defaultdict(list)
    prediction_table: list[pd.DataFrame] = []
    newly_processed_runs = []

    for run in tqdm(runs, desc="Processing runs"):
        try:
            logger.info(f"Processing run {run.name}")
            files = list(run.files())
            file_names = [f.name for f in files]

            task_results_table = get_wandb_table_as_df(
                files, "task_results_table", run.name
            )

            # This is to support legacy runs that have specific `diarization_prediction_table` before the more general `sample_results_table`
            if any(
                "diarization_prediction_table" in file_name for file_name in file_names
            ):
                prediction_table_name = "diarization_prediction_table"
            elif any("sample_results_table" in file_name for file_name in file_names):
                prediction_table_name = "sample_results_table"
            else:
                raise ValueError(
                    f"No prediction table found in run {run.name} - {file_names}"
                )
            diarization_prediction_table = get_wandb_table_as_df(
                files, prediction_table_name, run.name
            )

            task_results_tables = preprocess_task_results_table(task_results_table)
            diarization_prediction_table = preprocess_diarization_prediction_table(
                diarization_prediction_table
            )

            task_results_tables = {
                k: add_run_info(v, run) for k, v in task_results_tables.items()
            }
            diarization_prediction_table = add_run_info(
                diarization_prediction_table, run
            )

            prediction_table.append(diarization_prediction_table)

            for metric, df in task_results_tables.items():
                tasks_tables[metric].append(df)

            newly_processed_runs.append(run.id)
        except Exception as e:
            logger.error(f"Error processing run {run.name} - {run.id}: {e}")
            continue

    logger.info("Merging processed data")
    task_tables_merged: dict[str, pd.DataFrame] = {
        k: pd.concat(v, ignore_index=True) for k, v in tasks_tables.items()
    }
    prediction_table_merged = pd.concat(prediction_table, ignore_index=True)

    save_processed_data(task_tables_merged, prediction_table_merged, output_dir)

    # Update the processed runs file
    logger.info("Updating processed runs file")
    processed_runs.extend(newly_processed_runs)
    logger.info(
        f"Newly processed runs: {len(newly_processed_runs)}. Total number of processed runs: {len(processed_runs)}"
    )
    pd.DataFrame({"run_id": processed_runs}).to_csv(processed_runs_path, index=False)
    logger.info("Data processing complete")


def wandb_dag(
    project: str = "diarization-benchmarks",
    entity: str = "speakerkit",
    output_dir: str = "wandb_data",
    repo_id: str = "argmaxinc/interspeech-artifacts",
) -> None:
    # This function does the following:
    # 1. Downloads previously preprocessed data from the HF repo
    # 2. Preprocesses more data
    # 3. Pushes the preprocessed data to the repo
    logger.info(f"Starting data processing for project {project}")
    # Make sure output_dir exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Download the preprocessed data from the repo
    download_preprocessed_data(
        repo_id=repo_id, dir_to_download=output_dir, output_dir="."
    )
    # Preprocess more data
    wandb_data_processor(project=project, entity=entity, output_dir=output_dir)
    # Push the preprocessed data to the repo
    push_preprocessed_data(
        repo_id=repo_id, dir_to_push=output_dir, path_in_repo=output_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="diarization-benchmark")
    parser.add_argument("--entity", type=str, default="speakerkit")
    parser.add_argument("--output-dir", type=str, default="wandb_data")
    parser.add_argument(
        "--repo-id", type=str, default="argmaxinc/interspeech-artifacts"
    )
    args = parser.parse_args()
    wandb_dag(
        project=args.project,
        entity=args.entity,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
    )
