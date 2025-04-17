# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import unittest

import numpy as np

from sdbench.pipeline.diarization.coreml_pipeline.audio_chunker import AudioChunker

CHUNK_LENGTH = 30
WINDOW_LENGTH = 10
SAMPLE_RATE = 16_000


class TestAudioChunker(unittest.TestCase):
    def build_chunker(self, chunk_stride: int, window_stride: int) -> AudioChunker:
        return AudioChunker(
            chunk_length=CHUNK_LENGTH,
            chunk_stride=chunk_stride,
            window_length=WINDOW_LENGTH,
            window_stride=window_stride,
            sample_rate=SAMPLE_RATE,
            drop_outbound_windows=True,
        )

    def test_chunk_length_equals_audio_length(self) -> None:
        """Test when audio length equals chunk length."""
        chunk_strides = [21, 30]
        waveform = np.random.randn(CHUNK_LENGTH * SAMPLE_RATE)
        for chunk_stride in chunk_strides:
            chunker = self.build_chunker(chunk_stride=chunk_stride, window_stride=1)
            chunks, original_length, padded_length = chunker.chunk(waveform)
            self.assertEqual(
                len(chunks), 1, f"Expected 1 chunk for stride {chunk_stride}"
            )
            self.assertEqual(
                chunks[0].shape[0],
                original_length,
                "Chunk length should equal original length",
            )
            self.assertEqual(
                chunks[0].shape[0],
                padded_length,
                "Chunk length should equal padded length",
            )
            self.assertTrue(
                np.allclose(chunks[0], waveform), "Chunk should equal original waveform"
            )

    def test_audio_shorter_than_chunk(self) -> None:
        """Test when audio is shorter than chunk length."""
        waveform = np.random.randn((CHUNK_LENGTH - 10) * SAMPLE_RATE)
        chunker = self.build_chunker(chunk_stride=CHUNK_LENGTH, window_stride=1)
        chunks, original_length, padded_length = chunker.chunk(waveform)

        self.assertEqual(len(chunks), 1, "Should have one chunk")
        self.assertEqual(
            original_length, len(waveform), "Original length should match input"
        )
        self.assertEqual(
            padded_length, CHUNK_LENGTH * SAMPLE_RATE, "Should pad to chunk length"
        )
        self.assertTrue(
            np.allclose(chunks[0][:original_length], waveform),
            "Original audio should be preserved",
        )

    def test_overlapping_chunks(self) -> None:
        """Test overlapping chunks with different strides."""
        # Create a 50-second audio
        waveform = np.random.randn(50 * SAMPLE_RATE)

        # Test with 21s stride (should get 2 chunks: 0-30s and 21-51s)
        chunker = self.build_chunker(chunk_stride=21, window_stride=1)
        chunks, original_length, padded_length = chunker.chunk(waveform)

        self.assertEqual(len(chunks), 2, "Should have two chunks")
        self.assertEqual(
            original_length, len(waveform), "Original length should match input"
        )
        self.assertEqual(
            padded_length, 51 * SAMPLE_RATE, "Should pad to fit second chunk"
        )

        # Verify chunk contents
        self.assertTrue(
            np.allclose(chunks[0][: 30 * SAMPLE_RATE], waveform[: 30 * SAMPLE_RATE]),
            "First chunk should match first 30s",
        )
        self.assertTrue(
            np.allclose(chunks[1][: 29 * SAMPLE_RATE], waveform[21 * SAMPLE_RATE :]),
            "Second chunk should match from 21s onwards",
        )

    def test_exact_chunk_fit(self) -> None:
        """Test when audio length fits exactly with chunk size and stride."""
        # Create a 60-second audio (2 chunks of 30s with 30s stride)
        waveform = np.random.randn(60 * SAMPLE_RATE)
        chunker = self.build_chunker(chunk_stride=30, window_stride=1)
        chunks, original_length, padded_length = chunker.chunk(waveform)

        self.assertEqual(len(chunks), 2, "Should have two chunks")
        self.assertEqual(
            original_length, len(waveform), "Original length should match input"
        )
        self.assertEqual(padded_length, len(waveform), "Should not need padding")

        # Verify chunk contents
        self.assertTrue(
            np.allclose(chunks[0], waveform[: 30 * SAMPLE_RATE]),
            "First chunk should match first 30s",
        )
        self.assertTrue(
            np.allclose(chunks[1], waveform[30 * SAMPLE_RATE :]),
            "Second chunk should match last 30s",
        )

    def test_window_mask(self) -> None:
        """Test the window mask generation."""
        chunker = self.build_chunker(chunk_stride=30, window_stride=1)
        mask = chunker.get_windows_mask(60)  # 60 seconds of audio

        # Should have 2 chunks (0-30s and 30-60s)
        self.assertEqual(mask.shape[0], 2, "Should have 2 chunks")

        # Each chunk should have 21 windows
        self.assertEqual(mask.shape[1], 21, "Should have 21 windows per chunk")

        # All windows should be valid
        self.assertTrue(np.all(mask), "All windows should be valid")

    def test_window_mask_with_overlap(self) -> None:
        """Test window mask with overlapping chunks."""
        chunker = self.build_chunker(chunk_stride=21, window_stride=1)
        mask = chunker.get_windows_mask(50)  # 50 seconds of audio

        # Should have 2 chunks (0-30s and 21-51s)
        self.assertEqual(mask.shape[0], 2, "Should have 2 chunks")

        # Each chunk should have 21 windows
        self.assertEqual(mask.shape[1], 21, "Should have 21 windows per chunk")

        # All windows except the last one should be valid
        num_masks = mask.size
        expected_num_invalid_masks = 1
        expected_num_valid_masks = num_masks - expected_num_invalid_masks

        num_valid_masks = np.sum(mask).item()

        self.assertEqual(
            num_valid_masks,
            expected_num_valid_masks,
            "Should have the correct number of valid masks",
        )

    def test_audio_length_less_than_window_length(self) -> None:
        """Test when audio length is less than window length."""
        audio_length_seconds = 8.5
        audio_length_samples = int(audio_length_seconds * SAMPLE_RATE)
        waveform = np.random.randn(audio_length_samples)
        chunker = self.build_chunker(chunk_stride=21, window_stride=1)

        chunks, original_length, padded_length = chunker.chunk(waveform)
        masks = chunker.get_windows_mask(audio_length_seconds)

        # Should have 1 chunk
        self.assertEqual(
            len(chunks), 1, f"Should have one chunk, but got {len(chunks)}"
        )
        # Should only the first window should be valid
        nonzero_masks = masks[0, :, 0].nonzero()[0]
        self.assertEqual(
            len(nonzero_masks),
            1,
            f"Should have one valid window, but got {len(nonzero_masks)}",
        )
        self.assertEqual(
            nonzero_masks[0],
            0,
            f"Should have valid window at index 0, but got {nonzero_masks[0]}",
        )

    def test_mask_when_padding_is_needed(self) -> None:
        """Test when padding is needed."""

        # There are a few cases where we'll have to process windows with a little of padding reason e.g.
        # If audio_length is 31.9s; chunk_stride is 21s; window_stride is 1s; window_length is 10s
        # Our last window would have to be 22-32s to get the extra 0.9s of audio
        def check_mask_correctness(audio_length: float, window_stride: int) -> None:
            chunker = self.build_chunker(
                chunk_stride=CHUNK_LENGTH + window_stride - WINDOW_LENGTH,
                window_stride=window_stride,
            )
            mask, window_intervals = chunker.get_windows_mask(
                audio_length, return_intervals=True
            )
            flat_mask = mask.flatten()
            flat_window_intervals = window_intervals.reshape(-1, 2)

            valid_window_intervals = flat_window_intervals[flat_mask]
            # Second to last window should end before audio ends
            self.assertLess(
                valid_window_intervals[-2, 1].item(),
                audio_length,
                f"Second to last window should end before audio ends for audio length {audio_length} and window stride {window_stride}",
            )
            # Last window should be greater or equal to audio length
            self.assertGreaterEqual(
                valid_window_intervals[-1, 1].item(),
                audio_length,
                f"Last window should start after audio ends for audio length {audio_length} and window stride {window_stride}",
            )

        check_mask_correctness(31.9, 1)
        check_mask_correctness(33.9, 4)
        check_mask_correctness(63.9, 4)
        check_mask_correctness(34.9, 2)


if __name__ == "__main__":
    unittest.main()
