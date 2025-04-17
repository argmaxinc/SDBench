import warnings

warnings.filterwarnings("ignore")

import hydra
from argmaxtools.utils import get_logger
from omegaconf import OmegaConf

from sdbench.pipeline import PIPELINE_REGISTRY
from sdbench.runner import BenchmarkConfig, BenchmarkRunner

logger = get_logger(__name__)


@hydra.main(config_path="config", config_name="evaluation_config", version_base="1.1")
def main(config: OmegaConf) -> None:
    logger.info(
        f"Running benchmark with config:\n{OmegaConf.to_yaml(config, resolve=True)}"
    )

    benchmark_config = BenchmarkConfig(**config.benchmark_config)
    pipelines = [
        PIPELINE_REGISTRY[pipeline_name].from_dict(pipeline_config)
        for pipeline_name, pipeline_config in config.pipeline_configs.items()
    ]
    runner = BenchmarkRunner(benchmark_config, pipelines)
    benchmark_result = runner.run()


if __name__ == "__main__":
    main()
