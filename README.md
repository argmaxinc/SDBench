<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/sdbench-light.png">
  <source media="(prefers-color-scheme: light)" srcset="assets/sdbench-dark.png">
  <img alt="SDBench Logo" src="assets/sdbench-light.png">
</picture>

[![Paper](https://img.shields.io/badge/Paper-📄-blue)](http://argmaxinc.com/sdbench-paper)
[![Discord](https://img.shields.io/discord/1171912382512115722?style=flat&logo=discord&logoColor=969da4&label=Discord&labelColor=353a41&color=32d058&link=https%3A%2F%2Fdiscord.gg%2FG5F5GZGecC)](https://discord.gg/G5F5GZGecC)


`SDBench` is an open-source benchmarking tool for speaker diarization systems. The primary objective is to promote standardized, reproducible, and continuous evaluation of open-source and proprietary speaker diarization systems across on-device and server-side implementations.

Key features include:
- Simple interface to wrap your diarization, ASR, or ASR + diarization system
- Easily accessible and extensible metrics following `pyannote` standard metric implementations
- Modular and convenient configuration management through `hydra`
- Out-of-the-box `Weights & Biases` logging
- Availability of 13+ commonly used datasets (Original dataset license restrictions apply)

> [!TIP]
> Want to add your own diarization, ASR, or combined pipeline? Check out our [Adding a New Diarization Pipeline](#adding-a-new-diarization-pipeline) section for a step-by-step guide!

> [!IMPORTANT]
> Before getting started, please note that some datasets in our [Diarization Datasets](#diarization-datasets) section require special access or have license restrictions. While we provide dataset preparation utilities in `common/download_dataset`, you'll need to procure the raw data independently for these datasets. See the dataset table for details on access requirements.

## 🚀 Roadmap

- [ ] Distribute SpeakerKit CLI for reproduction
- [ ] Living Benchmark, running every other month

## Setting up the environment
<details>
<summary> Click to expand </summary>

In order to get started, first make sure you have `poetry` installed. The [official documentation](https://python-poetry.org/docs/#installing-with-the-official-installer) has instructions for how to install the `poetry` CLI.

If you already have `poetry` installed you can run `make setup` to install the dependencies and set up the environment.
If you use `conda` or `venv` directly to manage your python environment you can install poetry with `pip intstall poetry` and then run `make setup` to install the dependencies.

Example with `conda`:
```bash
conda create -n <your-env-name> python=3.11
conda activate <your-env-name>
pip install poetry
make setup
```
</details>

## Diarization Datasets
<details>
<summary> Click to expand </summary>

The benchmark suite uses several speaker diarization datasets that are stored on the HuggingFace Hub. You can find all the datasets used in our evaluation in this [collection](https://huggingface.co/collections/argmaxinc/diarization-datasets-67646304c9b5e2cf9720ec48). The datasets available in the aforementioned collection are:

| Dataset Name | Out-of-the-box | License | How to Access |
|-------------|--------------|----------|---------------|
| [earnings21](https://github.com/revdotcom/speech-datasets/tree/main/earnings21) | ✅ | CC BY-SA 4.0 | Provided |
| [msdwild](https://github.com/X-LANCE/MSDWILD/tree/master) | ❌ | [MSDWild License Agreement](https://github.com/X-LANCE/MSDWILD/blob/master/MSDWILD_license_agreement.pdf) | Use `common/download_dataset.py` script |
| [icsi-meetings](https://groups.inf.ed.ac.uk/ami/icsi/download/) | ✅ | CC BY 4.0 | Provided |
| [aishell-4](https://www.openslr.org/111/) | ✅ | CC BY-SA 4.0 | Provided |
| [ali-meetings](https://www.openslr.org/119/) | ✅ | CC BY-SA 4.0 | Provided |
| [voxconverse](https://github.com/joonson/voxconverse) | ✅ | CC BY 4.0 | Provided |
| [ava-avd](https://github.com/zcxu-eric/AVA-AVD/tree/main/dataset) | ✅ | MIT | Provided |
| [ami-sdm](https://groups.inf.ed.ac.uk/ami/corpus/) | ✅ | CC BY 4.0 | Provided |
| [ami-ihm](https://groups.inf.ed.ac.uk/ami/corpus/) | ✅ | CC BY 4.0 | Provided |
| [american-life-podcast](https://github.com/jovistos/TALAD) | ❌ | Not disclosed | Use `common/download_dataset.py` script |
| [dihard-III](https://catalog.ldc.upenn.edu/LDC2022S14) | ❌ | [LDC License Agreement](https://catalog.ldc.upenn.edu/license/ldc-non-members-agreement.pdf) | Request access to LDC and use `common/download_dataset.py` script to parse |
| [callhome](https://catalog.ldc.upenn.edu/LDC2001S97) | ❌ | [LDC License Agreement](https://catalog.ldc.upenn.edu/license/ldc-non-members-agreement.pdf) | Request access to LDC and use `common/download_dataset.py` script to parse |
| [ego-4d](https://ego4d-data.org/docs/start-here/) | ❌ | [Ego4D License Agreement](https://ego4ddataset.com/ego4d-license/) | Request access to Ego4D and use `common/download_dataset.py` script to parse |


From these datasets `voxconverse` and `ami` are not present as download options as they were already present in the HuggingFace Hub uploaded by [diarizers-community](https://huggingface.co/diarizers-community).

### Dataset Schema

The benchmark suite supports different types of pipelines (`Diarization`, `ASR`, and `Orchestration`) with varying schema requirements. All datasets must follow a base schema, with additional fields required for specific pipeline types.

#### Base Schema (Required for all pipelines)
- `audio`: Audio column containing:
  - `array`: Audio waveform as numpy array of shape `(n_samples,)`
  - `sampling_rate`: Sample rate as integer
- `timestamps_start`: List of `float` containing start timestamps of segments in seconds
- `timestamps_end`: List of `float` containing end timestamps of segments in seconds
- `speakers`: List of `str` containing speaker IDs for each segment

#### Additional Fields for Specific Pipeline Types

##### Diarization Pipeline
- `uem_timestamps`: Optional list of tuples `[(start, end), ...]` containing Universal Evaluation Map (UEM) timestamps for evaluation

##### ASR Pipeline
- `transcript`: List of strings containing the words in the transcript
- `word_timestamps`: Optional list of tuples `[(start, end), ...]` containing timestamps for each word
- `word_speakers`: Optional list of strings containing speaker IDs for each word

##### Orchestration Pipeline (Combined Diarization + ASR)
- All fields from both Diarization and ASR pipelines are required
- `word_speakers` must be provided if `word_timestamps` is present
- Length of `word_speakers` must match length of `transcript`
- Length of `word_timestamps` must match length of `transcript`

##### Validation Rules
- For ASR and Orchestration pipelines, if word-level information is provided:
  - `word_speakers` and `transcript` must have the same length
  - `word_timestamps` and `transcript` must have the same length
  - If `word_timestamps` is provided, `word_speakers` must also be provided

### Downloading Datasets

If you want to reproduce the exact dataset downloads and processing, you can use our dataset downloading scripts. First, make sure you have the required dependencies installed as mentioned in the `Getting Started` section and also install the `dataset` dependencies doing `poetry install --with dataset`

After installing the dependencies, you can run the dataset downloading script at `common/download_dataset.py`. For example, to download the ICSI meetings dataset, you can run:

```bash
poetry run python common/download_dataset.py --dataset icsi-meetings --hf-repo-owner <your-huggingface-username>
```

This will download the dataset and store locally at `raw_datasets/icsi-meetings` directory and upload it to the designated HuggingFace organization at `<your-huggingface-username>/icsi-meetings`. In case you only want to download and not push to HuggingFace, you can use the `--generate-only` flag.

For simplicity if you want to download all the datasets you can run:

```bash
# This will download all the datasets and store them in the raw_datasets directory
# Will not push to HuggingFace
make download-datasets
```

### NOTE:
- For datasets requiring Hugging Face access, make sure you have your `HF_TOKEN` environment variable set
- For the `American Life Podcast` dataset, you'll need Kaggle API credentials in `~/.kaggle/kaggle.json`
- For [`Callhome`](https://catalog.ldc.upenn.edu/LDC2001S97) and [`Dihard-III`](https://catalog.ldc.upenn.edu/LDC2022S14) you need to acquire the datasets from LDC first and then set their paths in the following env variables:
    - `DIHARD_DATASET_DIR` if not specified it will assume the directory lives at `~/third_dihard_challenge_eval/data`
    - `CALLHOME_AUDIO_ROOT` if not specified it will assume the directory lives at `~/callhome/nist_recognition_evaluation/r65_8_1/sid00sg1/data`
- The downloaded datasets will be stored in the `raw_datasets` directory (which is gitignored):

</details>

## Adding a New Diarization Pipeline

<details>
<summary> Click to expand </summary>

SDBench can be used as a library to evaluate your own diarization, transcription, or orchestration pipelines. The framework supports three types of pipelines:

1. **Diarization Pipeline**: For speaker diarization tasks
2. **Transcription Pipeline**: For ASR/transcription tasks
3. **Orchestration Pipeline**: For combined diarization and transcription tasks

### Creating Your Pipeline

1. Create a new Python file (e.g., `my_pipeline.py`) and implement your pipeline:

```python
from typing import Callable

from sdbench.dataset import DiarizationSample
from sdbench.pipeline.base import Pipeline, PipelineType, register_pipeline
from sdbench.pipeline.diarization.common import DiarizationOutput, DiarizationPipelineConfig
from sdbench.pipeline_prediction import DiarizationAnnotation

@register_pipeline
class MyDiarizationPipeline(Pipeline):
    _config_class = MyDiarizationConfig
    pipeline_type = PipelineType.DIARIZATION

    def build_pipeline(self) -> Callable[[dict], dict]:
        # Initialize your model/function and return a callable
        return my_diarizer_function

    def parse_input(self, input_sample: DiarizationSample) -> dict:
        # Convert DiarizationSample to your model's input format
        return {
            "waveform": input_sample.waveform,
            "sample_rate": input_sample.sample_rate
        }

    def parse_output(self, output: dict) -> DiarizationOutput:
        # Convert your model's output to DiarizationOutput
        return DiarizationOutput(prediction=annotation)
```

2. Create a configuration class for your pipeline:

```python
from pydantic import Field
from sdbench.pipeline.diarization.common import DiarizationPipelineConfig

class MyDiarizationConfig(DiarizationPipelineConfig):
    model_path: str = Field(..., description="Path to model weights")
    threshold: float = Field(0.5, description="Detection threshold")
    num_speakers: int | None = Field(None, description="Number of speakers (optional)")
```

3. Create a configuration file for your pipeline:

```yaml
# my_pipeline_config.yaml
out_dir: ./my_pipeline_logs
model_path: /path/to/model
threshold: 0.5
num_speakers: null
```

### Using Your Pipeline

1. Import your pipeline and create a benchmark configuration:

```python
from sdbench.runner import BenchmarkConfig, BenchmarkRunner, WandbConfig
from sdbench.metric import MetricOptions
from sdbench.dataset import DiarizationDatasetConfig

from my_pipeline import MyDiarizationPipeline, MyDiarizationConfig

# Create pipeline configuration
pipeline_config = MyDiarizationConfig(
    model_path="/path/to/model",
    threshold=0.5,
    num_speakers=None,
    out_dir="./my_pipeline_logs"
)

# Create benchmark configuration
benchmark_config = BenchmarkConfig(
    wandb_config=WandbConfig(
        project_name="my-diarization-benchmark",
        run_name="my-pipeline-evaluation",
        tags=["my-pipeline", "evaluation"],
        wandb_mode="online"  # or "offline" for local testing
    ),
    metrics={
        MetricOptions.DER: {},  # Diarization Error Rate
        MetricOptions.JER: {},  # Jaccard Error Rate
    },
    datasets={
        "voxconverse": DiarizationDatasetConfig(
            dataset_id="diarizers-community/voxconverse",
            split="test"
        )
    }
)

# Create pipeline instance
pipeline = MyDiarizationPipeline(pipeline_config)

# Create and run benchmark
runner = BenchmarkRunner(benchmark_config, [pipeline])
benchmark_result = runner.run()

print(benchmark_result.global_results[0])
```

2. For parallel processing, you can configure the number of worker processes in your pipeline config:

```python
pipeline_config = MyDiarizationConfig(
    model_path="/path/to/model",
    threshold=0.5,
    num_speakers=None,
    out_dir="./my_pipeline_logs",
    num_worker_processes=4,  # Number of parallel workers
    per_worker_chunk_size=2  # Samples per worker
)
```

3. To use Weights & Biases for experiment tracking, make sure to:
   - Set up your W&B account and get your API key
   - Make sure you're logged into your W&B account otherwise run `wandb login`
   - Configure the `wandb_config` in your benchmark configuration

The BenchmarkRunner will automatically:
- Run your pipeline on the specified datasets
- Calculate metrics for each sample
- Aggregate results globally
- Log everything to Weights & Biases (if configured)
- Handle parallel processing if enabled (specially interesting for APIs)
- Generate detailed reports and artifacts

### Pipeline Types and Requirements

#### Diarization Pipeline
- Must implement `build_pipeline()`, `parse_input()`, and `parse_output()`
- Input parsing should convert `DiarizationSample` to your model's expected format
- Output parsing should return a `DiarizationOutput` with a `prediction` field

#### Transcription Pipeline
- Must implement `build_pipeline()`, `parse_input()`, and `parse_output()`
- Input parsing should convert `DiarizationSample` to your model's expected format
- Output parsing should return a `TranscriptionOutput` with a `prediction` field

#### Orchestration Pipeline
- Must implement `build_pipeline()`, `parse_input()`, and `parse_output()`
- Can either:
  - Implement end-to-end diarization and transcription
  - Use `PostInferenceMergePipeline` to combine separate diarization and transcription pipelines
- Output parsing should return an `OrchestrationOutput` with a `prediction` field and optionaly `diarization` and `transcription` results


</details>

## Hydra Configuration
<details>
<summary> Click to expand </summary>

The benchmark suite uses Hydra for configuration management, providing a flexible and modular way to configure evaluation runs. The configuration files are organized in the following structure:

```
config
├── evaluation_config.yaml                      # Main evaluation configuration
├── benchmark_config                            # Base configurations for benchmarking
│   ├── datasets                                # Dataset-specific configs
│   ├── wandb_config                            # Weights & Biases logging configs
│   └── base.yaml                               # Default benchmark_config used in evaluation_config.yaml
└── pipeline_configs                            # Predefined pipeline configurations for ease of use
    ├── my_pipeline
    │   ├── base.yaml                           # Default config used in my_pipeline.yaml
    │   └── config
    │       ├── base.yaml                       # Default config used in MyPipeline
    │       └── diarization_config
    │           ├── chunking_config             # Defines different useful chunking configurations
    │           ├── cluster_definition          # Defines different useful cluster definitions
    │           ├── speaker_embedder_config     # Defines different useful speaker embedder configurations
    │           ├── speaker_segmenter_config    # Defines different useful speaker segmenter configurations
    │           └── base.yaml                   # Default diarization_config used in evaluation_config.yaml
    ├── my_pipeline.yaml                        # Uses MyPipeline as default pipeline
    └── pyannote.yaml                           # Defines configuration for PyAnnotePipeline
```

### Running Evaluations with Different Configurations

You can easily customize your evaluation runs using Hydra's override syntax. Here are some common usage patterns:

1. **Selecting Specific Pipelines**
```bash
# Run evaluation with only MyPipeline
poetry run python evaluation.py pipeline_configs=my_pipeline
```

2. **Modifying Pipeline Parameters**
You can override specific configuration parameters in two ways:

a. **Override by Value**:
```bash
# Change the speaker segmenter stride
poetry run python evaluation.py \
    pipeline_configs=my_pipeline \
    pipeline_configs.MyPipeline.config.diarization_config.speaker_segmenter_config.variant_name=stride_2
```

b. **Override by Config**:
```bash
# Use a predefined speaker segmenter configuration
poetry run python evaluation.py \
    pipeline_configs=my_pipeline \
    pipeline_configs/MyPipeline/config/diarization_config/speaker_segmenter_config=stride_2
```

Note: Use `-h` flag with any command to see the resulting configuration:
```bash
poetry run python evaluation.py pipeline_configs=my_pipeline -h
```
</details>
