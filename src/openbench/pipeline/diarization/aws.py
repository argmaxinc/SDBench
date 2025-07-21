# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import json
import os
import random
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable

import boto3
from botocore.exceptions import ClientError
from pyannote.audio.sample import Segment
from pydantic import BaseModel

from ...dataset import DiarizationSample
from ...pipeline_prediction import DiarizationAnnotation
from ..base import Pipeline, PipelineType, register_pipeline
from .common import DiarizationOutput, DiarizationPipelineConfig


__all__ = ["AWSTranscribePipeline", "AWSTranscribeConfig"]


class AWSTranscribeSegment(BaseModel):
    speaker: str
    start: float
    end: float


class AWSTranscribeDiarization(BaseModel):
    segments: list[AWSTranscribeSegment]

    def to_annotation(self) -> DiarizationAnnotation:
        annotation = DiarizationAnnotation()
        for segment in self.segments:
            annotation[Segment(segment.start, segment.end)] = segment.speaker
        return annotation.support()


class AWSTranscribeOutput(BaseModel):
    job_id: str
    status: str
    created_at: datetime
    completed_at: datetime
    diarization: AWSTranscribeDiarization

    def get_elapsed_time(self) -> float:
        return (self.completed_at - self.created_at).total_seconds()


class AWSTranscribeConfig(DiarizationPipelineConfig):
    bucket_name: str
    region_name: str


class AWSTranscribeAPI:
    TEMP_AUDIO_DIR = Path("audio_temp")

    def __init__(self, config: AWSTranscribeConfig):
        self.config = config
        self.max_retries = 10
        self.base_delay = 1  # Base delay in seconds
        self.max_delay = 64  # Maximum delay in seconds

    def _exponential_backoff(self, attempt: int) -> float:
        """
        Calculate delay with exponential backoff and jitter
        """
        delay = min(self.max_delay, self.base_delay * 2**attempt)
        return random.uniform(0, delay)

    def _with_retries(self, operation: Callable, *args, **kwargs):
        """
        Execute an AWS operation with retries and exponential backoff
        """
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return operation(*args, **kwargs)
            except ClientError as e:
                if e.response["Error"]["Code"] == "ThrottlingException":
                    last_exception = e
                    delay = self._exponential_backoff(attempt)
                    time.sleep(delay)
                    continue
                raise  # Re-raise if it's not a throttling error

        # If we've exhausted all retries
        raise last_exception

    def _get_clients(self):
        """
        Create new clients when needed - each process in the pool will need its own client.
        """
        transcribe_client = boto3.client("transcribe", region_name=self.config.region_name)
        s3_client = boto3.client("s3", region_name=self.config.region_name)
        return transcribe_client, s3_client

    def _upload_to_s3(self, audio_path: str) -> str:
        _, s3_client = self._get_clients()
        blob_name = os.path.basename(audio_path)
        s3_client.upload_file(
            Filename=audio_path,
            Bucket=self.config.bucket_name,
            Key=blob_name,
        )
        return blob_name

    def _transcribe_blob(self, blob_name: str) -> dict:
        """
        Runs a transcription job on the given blob and returns the results.
        The results are serialized in the pre-configured s3 bucket.
        """
        transcribe_client, s3_client = self._get_clients()
        job_name = uuid.uuid4().hex
        start_time = datetime.now()

        self._with_retries(
            transcribe_client.start_transcription_job,
            TranscriptionJobName=job_name,
            OutputBucketName=self.config.bucket_name,
            OutputKey=f"{blob_name}.json",
            Media={"MediaFileUri": f"s3://{self.config.bucket_name}/{blob_name}"},
            IdentifyMultipleLanguages=True,
            MediaFormat="wav",
            Settings={
                "ShowSpeakerLabels": True,
                "MaxSpeakerLabels": self.config.max_speakers,
            },
        )

        completed = False
        while not completed:
            try:
                response = self._with_retries(
                    transcribe_client.get_transcription_job,
                    TranscriptionJobName=job_name,
                )
                status = response["TranscriptionJob"]["TranscriptionJobStatus"]
                if status in ["COMPLETED", "FAILED"]:
                    completed = True
                    if status == "FAILED":
                        raise RuntimeError(f"Transcription job failed: {response}")
                else:
                    # Wait for 1 second before polling again
                    time.sleep(1)
            except ClientError as e:
                if e.response["Error"]["Code"] != "ThrottlingException":
                    raise
                # If throttled, back off and retry
                time.sleep(self._exponential_backoff(0))

        completed_at = datetime.now()

        # Get results
        temp_dir = tempfile.mkdtemp()
        result_path = os.path.join(temp_dir, f"{blob_name}.json")
        s3_client.download_file(
            Bucket=self.config.bucket_name,
            Key=f"{blob_name}.json",
            Filename=result_path,
        )

        with open(result_path) as f:
            transcript = json.load(f)

        # Clean up
        os.remove(result_path)

        # Parse segments
        segments = []
        for segment in transcript["results"]["speaker_labels"]["segments"]:
            segments.append(
                AWSTranscribeSegment(
                    speaker=segment["speaker_label"],
                    start=float(segment["start_time"]),
                    end=float(segment["end_time"]),
                )
            )

        return AWSTranscribeOutput(
            job_id=job_name,
            status="completed",
            created_at=start_time,
            completed_at=completed_at,
            diarization=AWSTranscribeDiarization(segments=segments),
        )

    def __call__(self, audio_path: str) -> AWSTranscribeOutput:
        blob_name = self._upload_to_s3(audio_path)
        return self._transcribe_blob(blob_name)


@register_pipeline
class AWSTranscribePipeline(Pipeline):
    _config_class = AWSTranscribeConfig
    pipeline_type = PipelineType.DIARIZATION

    TEMP_AUDIO_DIR = Path("audio_temp")

    def parse_input(self, input_sample: DiarizationSample) -> dict:
        audio_path = input_sample.save_audio(output_dir=self.TEMP_AUDIO_DIR)
        self._audio_path = audio_path
        return {"audio_path": str(audio_path)}

    def parse_output(self, output: AWSTranscribeOutput) -> DiarizationOutput:
        output = DiarizationOutput(
            prediction=output.diarization.to_annotation(),
            prediction_time=output.get_elapsed_time(),
        )
        # Remove audio from temp
        self._audio_path.unlink()
        return output

    def process_audio(self, input_sample):
        return self.api_client(input_sample["audio_path"])

    def build_diarizer(self) -> Callable:
        self.api_client = AWSTranscribeAPI(self.config)
        return self.process_audio
