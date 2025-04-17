import os
from contextlib import contextmanager
from pathlib import Path

from argmaxtools.utils import get_logger
from pyannote.metrics.base import BaseMetric

from .data_models import GlobalResult

logger = get_logger(__name__)


def get_global_results(
    metrics_dict: dict[str, BaseMetric],
    dataset_name: str,
    pipeline_name: str,
) -> list[GlobalResult]:
    global_results = []
    for metric_name, metric in metrics_dict.items():
        # This is how you get the global result of a metric in pyannote
        global_result = abs(metric)
        # Get the global result of the metric components
        detailed_result = {
            component: metric[component] for component in metric.components_
        }
        # This is how you get the confidence interval of a metric in pyannote
        # confidence interval is computed with `scipy.stats.bayes_mvs` taking only the interval for the mean
        avg_result, (lower_bound, upper_bound) = metric.confidence_interval(alpha=0.9)
        global_results.append(
            GlobalResult(
                dataset_name=dataset_name,
                pipeline_name=pipeline_name,
                metric_name=metric_name,
                global_result=global_result,
                detailed_result=detailed_result,
                avg_result=avg_result,
                upper_bound=upper_bound,
                lower_bound=lower_bound,
            )
        )

    return global_results


@contextmanager
def change_directory(path: Path | str):
    """Context manager for changing the current working directory.
    If the path is a string, it will be converted to a Path object.
    If the path is not a directory, it will be created.
    """
    if isinstance(path, str):
        path = Path(path)

    prev_dir = Path.cwd()
    try:
        logger.info(f"Creating directory {path}")
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Changing directory to {path}")
        os.chdir(path)
        yield
    finally:
        os.chdir(prev_dir)
