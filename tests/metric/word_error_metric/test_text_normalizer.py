# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.

import unittest

from sdbench.metric.word_error_metrics.text_normalizer import EnglishTextNormalizer


class TestTextNormalizer(unittest.TestCase):
    def test_text_normalizer_with_and_without_speakers(self):
        # Test cases with various text patterns that should be normalized
        test_cases = [
            {
                "words": ["I", "can't", "believe", "it's", "not", "butter"],
                "speakers": ["A", "A", "A", "A", "A", "A"],
            },
            {
                "words": ["Let's", "go", "to", "the", "park", "at", "2pm"],
                "speakers": ["A", "A", "A", "A", "A", "A", "A"],
            },
            {
                "words": ["I", "wanna", "go", "home", "now"],
                "speakers": ["A", "A", "A", "A", "A"],
            },
            {
                "words": ["The", "colour", "is", "grey"],
                "speakers": ["A", "A", "A", "A"],
            },
            {
                "words": ["Hello", "how", "are", "you", "I'm", "good", "thanks"],
                "speakers": ["A", "A", "A", "A", "B", "B", "B"],
            },
        ]

        normalizer = EnglishTextNormalizer()

        for test_case in test_cases:
            # Test without speakers
            words_without_speakers, _ = normalizer(test_case["words"], None)

            # Test with speakers
            words_with_speakers, _ = normalizer(
                test_case["words"], test_case["speakers"]
            )

            # Verify both methods yield the same normalized text
            self.assertEqual(
                words_without_speakers,
                words_with_speakers,
                f"Normalization with and without speakers differs for input: {test_case['words']}\n"
                f"Without speakers: {words_without_speakers}\n"
                f"With speakers: {words_with_speakers}",
            )


if __name__ == "__main__":
    unittest.main()
