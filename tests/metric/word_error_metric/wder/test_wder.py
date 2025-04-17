# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.

import os
import random
import subprocess
import unittest

from argmaxtools.utils import get_logger

from sdbench.metric import WordDiarizationErrorRate
from sdbench.pipeline_prediction import Transcript

logger = get_logger(__name__)

RANDOM_SEED = 69


class ReferenceWDER:
    """Wrapper class for running the reference WDER implementation through Docker."""

    def __init__(self) -> None:
        self.docker_image = "reference-wder:latest"
        self._build_docker_image()

    def _build_docker_image(self) -> None:
        """Build the Docker image if it doesn't exist."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"Building Docker image in {script_dir}")
        subprocess.run(
            ["docker", "build", "-t", self.docker_image, "-f", "Dockerfile", "."],
            cwd=script_dir,
            check=True,
        )
        logger.info("Docker image built successfully")

    def _format_input(self, transcript: Transcript) -> tuple[str, str]:
        """Format a list of Words into text and speaker strings."""
        text = transcript.get_transcript_string()
        speakers = transcript.get_speakers_string()
        return text, speakers

    def __call__(self, reference: Transcript, hypothesis: Transcript) -> float:
        """Compute WDER between reference and hypothesis using the reference implementation."""
        ref_text, ref_spk = self._format_input(reference)
        hyp_text, hyp_spk = self._format_input(hypothesis)

        # Run the Docker container with the inputs
        logger.info(
            f"Running Docker container with inputs: {ref_text}, {ref_spk}, {hyp_text}, {hyp_spk}"
        )
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                self.docker_image,
                "--reference-text",
                ref_text,
                "--reference-speaker",
                ref_spk,
                "--hypothesis-text",
                hyp_text,
                "--hypothesis-speaker",
                hyp_spk,
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse the output (which should be just the WDER value)
        return float(result.stdout.strip())


def create_transcript(text: str, speakers: str) -> Transcript:
    """Create a list of Words from text and speaker strings."""
    return Transcript.from_words_info(
        words=text.split(),
        start=None,
        end=None,
        speaker=speakers.split(),
    )


class TestWDER(unittest.TestCase):
    def setUp(self) -> None:
        self.wder = WordDiarizationErrorRate()
        self.reference_wder = ReferenceWDER()

    def tearDown(self) -> None:
        self.wder = None
        self.reference_wder = None

    def _compute_and_compare(
        self,
        reference_text: str,
        reference_speakers: str,
        hypothesis_text: str,
        hypothesis_speakers: str,
    ) -> None:
        reference = create_transcript(reference_text, reference_speakers)
        hypothesis = create_transcript(hypothesis_text, hypothesis_speakers)

        main_result = self.wder(reference=reference, hypothesis=hypothesis)
        ref_result = self.reference_wder(reference=reference, hypothesis=hypothesis)
        self.assertEqual(
            main_result,
            ref_result,
            f"Main result: {main_result}, Reference result: {ref_result}",
        )

    def test_wder_perfect_match(self) -> None:
        # Reference text-speakers:
        reference_text = "a b c d e f g h i j"
        reference_speakers = "1 1 1 1 2 2 2 2 2"
        # Hypothesis text-speakers:
        hypothesis_text = "a b c d e f g h i j"
        hypothesis_speakers = "1 1 1 1 2 2 2 2 2"

        self._compute_and_compare(
            reference_text=reference_text,
            reference_speakers=reference_speakers,
            hypothesis_text=hypothesis_text,
            hypothesis_speakers=hypothesis_speakers,
        )

    def test_wder_completely_wrong_speaker(self) -> None:
        # Reference text-speakers:
        reference_text = "a b c d e f g h i j"
        reference_speakers = "1 1 1 1 2 2 2 2 2"
        # Hypothesis text-speakers:
        hypothesis_text = "a b c d e f g h i j"
        hypothesis_speakers = "3 3 3 3 3 3 3 3 3 3"

        self._compute_and_compare(
            reference_text=reference_text,
            reference_speakers=reference_speakers,
            hypothesis_text=hypothesis_text,
            hypothesis_speakers=hypothesis_speakers,
        )

    def test_wder_completely_wrong_text(self) -> None:
        # Reference text-speakers:
        reference_text = "a b c d e f g h i j"
        reference_speakers = "1 1 1 1 2 2 2 2 2"
        # Hypothesis text-speakers:
        hypothesis_text = "l l l l l l l l l l"
        hypothesis_speakers = "1 1 1 1 2 2 2 2 2"

        self._compute_and_compare(
            reference_text=reference_text,
            reference_speakers=reference_speakers,
            hypothesis_text=hypothesis_text,
            hypothesis_speakers=hypothesis_speakers,
        )

    def test_wder_partially_wrong_text(self) -> None:
        # Reference text-speakers:
        reference_text = "a b c d e f g h i j"
        reference_speakers = "1 1 1 1 2 2 2 2 2"
        # Hypothesis text-speakers:
        hypothesis_text = "a b c d x f g h i j"
        hypothesis_speakers = "1 1 1 1 2 2 2 2 2"

        self._compute_and_compare(
            reference_text=reference_text,
            reference_speakers=reference_speakers,
            hypothesis_text=hypothesis_text,
            hypothesis_speakers=hypothesis_speakers,
        )

    def test_wder_partially_wrong_speaker(self) -> None:
        # Reference text-speakers:
        reference_text = "a b c d e f g h i j"
        reference_speakers = "1 1 1 1 2 2 2 2 2"
        # Hypothesis text-speakers:
        hypothesis_text = "a b c d e f g h i j"
        hypothesis_speakers = "1 1 1 1 3 3 3 3 3"

        self._compute_and_compare(
            reference_text=reference_text,
            reference_speakers=reference_speakers,
            hypothesis_text=hypothesis_text,
            hypothesis_speakers=hypothesis_speakers,
        )

    def test_wder_text_deletion(self) -> None:
        # Reference text-speakers:
        reference_text = "a b c d e f g h i j"
        reference_speakers = "1 1 1 1 2 2 2 2 2"
        # Hypothesis text-speakers:
        hypothesis_text = "a b c d e f g h"
        hypothesis_speakers = "1 1 1 1 2 2 2"

        self._compute_and_compare(
            reference_text=reference_text,
            reference_speakers=reference_speakers,
            hypothesis_text=hypothesis_text,
            hypothesis_speakers=hypothesis_speakers,
        )

    def test_wder_text_insertion(self) -> None:
        # Reference text-speakers:
        reference_text = "a b c d e f g h i j"
        reference_speakers = "1 1 1 1 2 2 2 2 2"
        # Hypothesis text-speakers:
        hypothesis_text = "a b c d e f g h i j k"
        hypothesis_speakers = "1 1 1 1 2 2 2 2 2"

        self._compute_and_compare(
            reference_text=reference_text,
            reference_speakers=reference_speakers,
            hypothesis_text=hypothesis_text,
            hypothesis_speakers=hypothesis_speakers,
        )

    def test_wder_less_speakers(self) -> None:
        # Reference text-speakers:
        reference_text = "a b c d e f g h i j"
        reference_speakers = "1 1 1 1 2 2 3 4 5"
        # Hypothesis text-speakers:
        hypothesis_text = "a b c d e f g h i j"
        hypothesis_speakers = "1 1 1 1 2 2 2 2 2"

        self._compute_and_compare(
            reference_text=reference_text,
            reference_speakers=reference_speakers,
            hypothesis_text=hypothesis_text,
            hypothesis_speakers=hypothesis_speakers,
        )

    def test_wder_more_speakers(self) -> None:
        # Reference text-speakers:
        reference_text = "a b c d e f g h i j"
        reference_speakers = "1 1 1 1 2 2 2 2 2"
        # Hypothesis text-speakers:
        hypothesis_text = "a b c d e f g h i j"
        hypothesis_speakers = "1 1 1 1 2 2 2 2 2 3"

        self._compute_and_compare(
            reference_text=reference_text,
            reference_speakers=reference_speakers,
            hypothesis_text=hypothesis_text,
            hypothesis_speakers=hypothesis_speakers,
        )

    def test_wder_random_modifications(self) -> None:
        # Reference text-speakers:
        reference_text = "a b c d e f g h i j"
        reference_speakers = "1 1 1 1 2 2 2 2 2"

        # Set random seed
        random.seed(RANDOM_SEED)

        # Run multiple iterations
        for iteration in range(10):  # Run 10 different random modifications
            # Create hypothesis with random modifications
            words = reference_text.split()
            speakers = reference_speakers.split()
            hypothesis_words = words.copy()
            hypothesis_speakers = speakers.copy()

            # For each position, randomly choose whether to modify it and how
            for pos in range(min(len(hypothesis_words), len(hypothesis_speakers))):
                if random.random() < 0.5:  # 50% chance to modify each position
                    modification_type = random.choice(["text", "speaker", "both"])

                    if modification_type in ["text", "both"]:
                        hypothesis_words[pos] = "x"
                    if modification_type in ["speaker", "both"]:
                        hypothesis_speakers[pos] = str(random.randint(1, 5))

            hypothesis_text = " ".join(hypothesis_words)
            hypothesis_speakers = " ".join(hypothesis_speakers)

            self._compute_and_compare(
                reference_text=reference_text,
                reference_speakers=reference_speakers,
                hypothesis_text=hypothesis_text,
                hypothesis_speakers=hypothesis_speakers,
            )
