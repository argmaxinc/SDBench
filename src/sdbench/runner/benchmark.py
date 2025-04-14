import os
from multiprocessing import Pool
from pathlib import Path
from typing import Any, NamedTuple

import tqdm
from argmaxtools.utils import get_logger
from pyannote.metrics.base import BaseMetric

import wandb

from ..dataset import DiarizationDataset, DiarizationSample
from ..metric import MetricRegistry
from ..pipeline import Pipeline, PipelineType
from .config import BenchmarkConfig
from .data_models import (
    BenchmarkResult,
    DiarizationSampleResult,
    GlobalResult,
    TaskResult,
    TranscriptionSampleResult,
)
from .utils import change_directory, get_global_results
from .wandb_logger import DiarizationWandbLogger, TranscriptionWandbLogger

logger = get_logger(__name__)


class ProcessingResult(NamedTuple):
    sample_result: DiarizationSampleResult | TranscriptionSampleResult
    task_results: list[TaskResult]
    sample_id: int
    metrics_string: str


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig, pipelines: list[Pipeline]):
        """Runs benchmarks for diarization pipelines.

        This class handles:
        - Pipeline execution on multiple datasets
        - Metric calculation and aggregation
        - Parallel processing support
        - Wandb logging

        Args:
            config: Benchmark configuration
            pipelines: List of pipelines to benchmark
        """
        self.config = config
        self.pipelines = pipelines
        self.logger_map = {
            PipelineType.DIARIZATION: DiarizationWandbLogger,
            PipelineType.TRANSCRIPTION: TranscriptionWandbLogger,
            PipelineType.ORCHESTRATION: TranscriptionWandbLogger,
        }

    def _get_metrics(self, pipeline: Pipeline) -> dict[str, BaseMetric]:
        metrics_dict = {}
        available_metrics = MetricRegistry.get_available_metrics(pipeline.pipeline_type)
        for metric_name, kwargs in self.config.metrics.items():
            if metric_name not in available_metrics:
                continue
            metrics_dict[metric_name] = MetricRegistry.get_metric(metric_name, **kwargs)
        return metrics_dict

    def _process_single_sample_wrapper(self, args):
        """
        Wrapper function to unpack arguments correctly
        """
        sample_and_id, pipeline, dataset_name, metrics_dict, dataset_length = args
        return self._process_single_sample(
            sample_and_id, pipeline, dataset_name, metrics_dict, dataset_length
        )

    def _process_single_sample(
        self,
        sample_and_id: tuple[int, DiarizationSample],
        pipeline: Pipeline,
        dataset_name: str,
        metrics_dict: dict,
        dataset_length: int,
    ) -> ProcessingResult:
        sample_id, sample = sample_and_id
        # Run pipeline
        output = pipeline(sample)
        prediction_time = output.prediction_time
        audio_duration = sample.get_audio_duration()

        # Create sample result based on pipeline type
        if pipeline.pipeline_type == PipelineType.DIARIZATION:
            sample_result = DiarizationSampleResult(
                dataset_name=dataset_name,
                sample_id=sample_id,
                pipeline_name=pipeline.__class__.__name__,
                prediction=output.prediction,
                embeddings=output.embeddings,
                cluster_labels=output.cluster_labels,
                centroids=output.centroids,
                prediction_time=prediction_time,
                audio_duration=audio_duration,
                num_speakers_predicted=len(output.prediction.labels()),
                num_speakers_reference=len(sample.annotation.labels()),
            )
        elif pipeline.pipeline_type in [
            PipelineType.TRANSCRIPTION,
            PipelineType.ORCHESTRATION,
        ]:
            sample_result = TranscriptionSampleResult(
                dataset_name=dataset_name,
                sample_id=sample_id,
                pipeline_name=pipeline.__class__.__name__,
                prediction=output.prediction,
                prediction_time=prediction_time,
                audio_duration=audio_duration,
            )
        else:
            raise ValueError(f"Unsupported pipeline type: {pipeline.pipeline_type}")

        # Process metrics
        task_results = []
        metrics_logging_string = ""

        for metric_name, metric in metrics_dict.items():
            # The metric returns a dictionary that is also stored in the metric object as a state to compute the global result
            # We copy to avoid any side effects that may happen while interacting with dictionary for reporting
            reference = (
                sample.annotation
                if pipeline.pipeline_type == PipelineType.DIARIZATION
                else sample.transcript
            )
            _metric_output = metric(
                hypothesis=output.prediction,
                reference=reference,
                uem=sample.uem,
                detailed=True,
            )
            metric_output = _metric_output.copy()

            detailed_result = {
                component_name: component_value
                for component_name, component_value in metric_output.items()
                if component_name != metric.name
            }
            result = metric_output[metric.name]

            task_results.append(
                TaskResult(
                    dataset_name=dataset_name,
                    sample_id=sample_id,
                    pipeline_name=pipeline.__class__.__name__,
                    metric_name=metric_name,
                    result=result,
                    detailed_result=detailed_result,
                )
            )
            metrics_logging_string += f"{metric_name} - Sample: {result:4g}\n{metric_name} - Global: {abs(metric):4g}\n"

        # Create logging string
        logging_string = (
            "\n=========================================================\n"
            f"Pipeline: {pipeline.__class__.__name__}\n"
            f"Dataset: {dataset_name}\n"
            f"Iteration: {sample_id + 1} of {dataset_length} ({((sample_id + 1) / dataset_length):.2%})\n"
            f"Prediction time: {prediction_time:.4g} seconds\n"
            f"Audio duration: {audio_duration:.4g} seconds\n"
            f"Speed Factor: {audio_duration / prediction_time:.4g}x\n"
            "---------------------------------------------------------\n"
            "Metrics:\n"
            f"{metrics_logging_string}"
            "---------------------------------------------------------\n"
            "=========================================================\n"
        )

        return ProcessingResult(sample_result, task_results, sample_id, logging_string)

    def _run_pipeline_on_dataset_parallel(
        self,
        pipeline: Pipeline,
        dataset: DiarizationDataset,
        dataset_name: str,
    ) -> tuple[
        list[DiarizationSampleResult | TranscriptionSampleResult],
        list[TaskResult],
        list[GlobalResult],
    ]:
        """
        Parallel version of _run_pipeline_on_dataset using multiprocessing.

        Args:
            pipeline: Pipeline to evaluate
            dataset: Dataset to evaluate on
            dataset_name: Name of the dataset
        Returns:
            Tuple of (sample_results, task_results, global_results)

        NOTE: imap (similarly to map but uses lazy evaluation) will chop the iterable into
            chunks based on the (per_worker_chunk_size parameter) and submit them to the worker processes as separate tasks.
            A rule of thumb while setting this parameter is to base it on the number of
            worker processes and the number of samples in the dataset.
            If you spawn many worker processes, it's best you keep the chunk size small (you can keep it 1).
            On the contrary, if you spawn fewer worker processes, each one can process larger chunks of data
            without risking the overhead of inter-process communication.

        Ref: https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.imap
        """
        metrics_dict = self._get_metrics(pipeline)
        dataset_length = len(dataset)

        args_list = [
            ((i, sample), pipeline, dataset_name, metrics_dict, dataset_length)
            for i, sample in enumerate(dataset)
        ]

        # NOTE: Currently, pipelines that utilize the MPS backend are not supported in parallel mode.
        # This is due to the limitation of sharing tensors across processes.
        # As workaround would be to move tensors to the CPU before processing in separate processes,
        # but this would defeat the purpose of using the MPS backend and would be slower.
        # Ref: https://github.com/pytorch/pytorch/issues/87688
        with Pool(processes=pipeline.config.num_worker_processes) as pool:
            results = list(
                tqdm.tqdm(
                    pool.imap(
                        self._process_single_sample_wrapper,
                        args_list,
                        chunksize=pipeline.config.per_worker_chunk_size,
                    ),
                    total=dataset_length,
                    desc=f"Processing {dataset_name}",
                )
            )

        # Sort results by sample_id to maintain order
        results.sort(key=lambda x: x.sample_id)

        # Separate results
        per_sample_results = [r.sample_result for r in results]
        per_task_results = [task for r in results for task in r.task_results]

        # Log results
        for result in results:
            logger.info(result.metrics_string)

        # Now update the metrics in the main process
        for metric in metrics_dict.values():
            # Clear any existing state
            metric.reset()
            # Update metric with all results
            for sample_result in per_sample_results:
                metric(
                    hypothesis=sample_result.prediction,
                    reference=dataset[sample_result.sample_id].annotation,
                    uem=dataset[sample_result.sample_id].uem,
                    detailed=True,
                )

        # Calculate global results after updating metrics
        global_results = get_global_results(
            metrics_dict=metrics_dict,
            dataset_name=dataset_name,
            pipeline_name=pipeline.__class__.__name__,
        )

        return per_sample_results, per_task_results, global_results

    def _run_pipeline_on_dataset(
        self,
        pipeline: Pipeline,
        dataset: DiarizationDataset,
        dataset_name: str,
    ) -> tuple[
        list[DiarizationSampleResult | TranscriptionSampleResult],
        list[TaskResult],
        list[GlobalResult],
    ]:
        per_sample_results: list[
            DiarizationSampleResult | TranscriptionSampleResult
        ] = []
        per_task_results: list[TaskResult] = []

        metrics_dict = self._get_metrics(pipeline)
        dataset_length = len(dataset)

        for sample_id, sample in enumerate(dataset):
            processing_result = self._process_single_sample(
                sample_and_id=(sample_id, sample),
                pipeline=pipeline,
                dataset_name=dataset_name,
                metrics_dict=metrics_dict,
                dataset_length=dataset_length,
            )
            per_sample_results.append(processing_result.sample_result)
            per_task_results.extend(processing_result.task_results)

            logger.info(processing_result.metrics_string)

        global_results = get_global_results(
            metrics_dict=metrics_dict,
            dataset_name=dataset_name,
            pipeline_name=pipeline.__class__.__name__,
        )
        return per_sample_results, per_task_results, global_results

    def run(self) -> BenchmarkResult:
        """
        Run the benchmark on the given datasets.
        """
        per_sample_results: list[
            DiarizationSampleResult | TranscriptionSampleResult
        ] = []
        per_task_results: list[TaskResult] = []
        per_dataset_global_results: list[GlobalResult] = []

        # Getting config to log to wandb
        wandb_config = self.config.get_wandb_config_to_log()
        for pipeline in self.pipelines:
            # Get logger
            wandb_logger = self.logger_map[pipeline.pipeline_type](
                output_dir=pipeline.config.out_dir
            )
            # Add pipeline info to wandb config
            wandb_config["pipeline_name"] = pipeline.__class__.__name__
            wandb_config["pipeline_config"] = pipeline.config.model_dump()

            with change_directory(pipeline.config.out_dir), wandb.init(
                project=self.config.wandb_config.project_name,
                name=self.config.wandb_config.run_name,
                mode=self.config.wandb_config.wandb_mode,
                tags=(
                    self.config.wandb_config.tags + [pipeline.__class__.__name__]
                    if self.config.wandb_config.tags
                    else [pipeline.__class__.__name__]
                ),
                config=wandb_config,
            ) as run:
                for dataset_name, dataset_config in self.config.datasets.items():
                    ds = DiarizationDataset.from_config(dataset_config)

                    logger.info(
                        f"Evaluating {pipeline.__class__.__name__} on {dataset_name}..."
                    )

                    if pipeline.config.num_worker_processes:
                        logger.info(
                            f"Executing in parallel mode with {pipeline.config.num_worker_processes} workers"
                        )
                        results = self._run_pipeline_on_dataset_parallel(
                            pipeline,
                            ds,
                            dataset_name,
                        )
                    else:
                        logger.info("Executing in sequential mode")
                        results = self._run_pipeline_on_dataset(
                            pipeline, ds, dataset_name
                        )

                    sample_results, task_results, global_results = results
                    wandb_logger(global_results, task_results, sample_results)

                    per_sample_results.extend(sample_results)
                    per_task_results.extend(task_results)
                    per_dataset_global_results.extend(global_results)

        return BenchmarkResult(
            sample_results=per_sample_results,
            task_results=per_task_results,
            global_results=per_dataset_global_results,
        )
