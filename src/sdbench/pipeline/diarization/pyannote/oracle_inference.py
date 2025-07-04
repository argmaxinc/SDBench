# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from pathlib import Path
from typing import Callable, Optional, Text, Tuple, Union

import numpy as np
import torch
from pyannote.audio.core.inference import Inference
from pyannote.audio.core.io import Audio, AudioFile
from pyannote.audio.core.model import Model, Specifications
from pyannote.audio.core.task import Resolution
from pyannote.audio.utils.multi_task import map_with_specifications
from pyannote.audio.utils.reproducibility import fix_reproducibility
from pyannote.core import Segment, SlidingWindow, SlidingWindowFeature

from ....pipeline_prediction import DiarizationAnnotation


class OracleSegmenterInference(Inference):
    """Oracle segmenter inference that uses ground truth annotations."""

    def __init__(
        self,
        model: Union[Model, Text, Path],
        window: Text = "sliding",
        duration: Optional[float] = None,
        step: Optional[float] = None,
        pre_aggregation_hook: Callable[[np.ndarray], np.ndarray] = None,
        skip_aggregation: bool = False,
        skip_conversion: bool = False,
        device: Optional[torch.device] = None,
        batch_size: int = 32,
        use_auth_token: Union[Text, None] = None,
    ):
        super().__init__(
            model,
            window,
            duration,
            step,
            pre_aggregation_hook,
            skip_aggregation,
            skip_conversion,
            device,
            batch_size,
            use_auth_token,
        )

    def __call__(
        self,
        file: AudioFile,
        hook: Optional[Callable] = None,
        job_id: Optional[str] = None,
    ) -> Union[
        Tuple[Union[SlidingWindowFeature, np.ndarray]],
        Union[SlidingWindowFeature, np.ndarray],
    ]:
        """Run inference using ground truth annotations.

        Parameters
        ----------
        file : AudioFile
            Audio file with ground truth annotations.
        hook : callable, optional
            Progress callback function.

        Returns
        -------
        output : SlidingWindowFeature or np.ndarray
            Model output based on ground truth annotations.
        """
        fix_reproducibility(self.device)

        waveform, sample_rate = self.model.audio(file)
        duration = Audio(mono="downmix").get_duration(file)
        if self.window == "sliding":
            return self.slide(hook=hook, reference=file["annotation"], duration=duration)

        outputs: Union[np.ndarray, Tuple[np.ndarray]] = self.infer(waveform[None])

        def __first_sample(outputs: np.ndarray, **kwargs) -> np.ndarray:
            return outputs[0]

        return map_with_specifications(self.model.specifications, __first_sample, outputs)

    def slide(
        self,
        hook: Optional[Callable],
        reference: DiarizationAnnotation,
        duration: float,
    ) -> Union[SlidingWindowFeature, Tuple[SlidingWindowFeature]]:
        """Slide over audio chunks using ground truth annotations.

        Parameters
        ----------
        hook : callable, optional
            Progress callback function.
        reference : DiarizationAnnotation
            Ground truth annotations.
        duration : float
            Audio duration.

        Returns
        -------
        SlidingWindowFeature
            Segmentation based on ground truth annotations.
        """

        def random_spkr_assigment(arr):
            if arr.shape[1] <= 3:
                return arr.copy()
            # Keep most active 3 speakers
            result = arr[:, :3].copy()
            # Reassign remaining speakers starting from index 3
            for col in range(3, arr.shape[1]):
                # Randomly select target speaker (0, 1, or 2)
                target_col = np.random.randint(0, 3)
                # Assign Least Talkative Speaker to Randomly Selected Target Speaker
                result[:, target_col] += arr[:, col]

            return result

        def get_frames(receptive_field, specifications: Optional[Specifications] = None) -> SlidingWindow:
            if specifications.resolution == Resolution.CHUNK:
                return SlidingWindow(start=0.0, duration=self.duration, step=self.step)
            return receptive_field

        frames: Union[SlidingWindow, Tuple[SlidingWindow]] = map_with_specifications(
            self.model.specifications, get_frames, self.model.receptive_field
        )

        # Get speaker labels and handle maximum number of speakers
        labels = reference.labels()
        actual_num_speakers = len(labels)
        num_speakers = 3  # Hard-coded maximum number of speakers

        window = SlidingWindow(start=0.0, duration=self.duration, step=self.step)
        segmentations = []

        for chunk in window(Segment(0.0, duration)):
            chunk_segmentation: SlidingWindowFeature = reference.discretize(
                chunk,
                resolution=frames,
                labels=labels,
                duration=window.duration,
            )

            if num_speakers < actual_num_speakers:
                # Keep only the most talkative speakers
                most_talkative_index = np.argsort(np.sum(chunk_segmentation, axis=0))[::-1]
                chunk_segmentation = chunk_segmentation[:, most_talkative_index]
                # Randomly Assign Least Talkative Speakers if Active Speakers > 3
                active_speaker_count = (chunk_segmentation.sum(axis=0) > 0).sum()
                (
                    segmentations.append(random_spkr_assigment(chunk_segmentation))
                    if active_speaker_count >= num_speakers
                    else segmentations.append(chunk_segmentation[:, :num_speakers])
                )

            else:
                # Zero pad to have 3 speakers
                segmentations.append(
                    np.pad(
                        chunk_segmentation.data,
                        ((0, 0), (0, num_speakers - chunk_segmentation.data.shape[1])),
                        "constant",
                        constant_values=0,
                    )
                )

        return SlidingWindowFeature(np.float32(np.stack(segmentations)), window)
