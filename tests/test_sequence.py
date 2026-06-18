import unittest

import numpy as np

from guided_streaming_policy.core.sequence import (
    SequencePointer,
    create_sequence_pointers,
    extract_sequence,
    get_data_stats,
    normalize_data,
    unnormalize_data,
)


class SequenceTests(unittest.TestCase):
    def test_create_sequence_pointers_with_padding(self) -> None:
        pointers = create_sequence_pointers(
            episode_ends=np.array([3]),
            sequence_length=2,
            pad_before=1,
            pad_after=0,
        )

        self.assertEqual(len(pointers), 3)
        self.assertEqual(
            pointers[0],
            SequencePointer(
                buffer_start_idx=0,
                buffer_end_idx=1,
                sample_start_idx=1,
                sample_end_idx=2,
            ),
        )

    def test_extract_sequence_applies_boundary_padding(self) -> None:
        train_data = {
            "action": np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        }
        pointer = SequencePointer(
            buffer_start_idx=0,
            buffer_end_idx=2,
            sample_start_idx=1,
            sample_end_idx=3,
        )

        result = extract_sequence(train_data, sequence_length=4, ptr=pointer)
        np.testing.assert_allclose(result["action"], np.array([[1.0], [1.0], [2.0], [2.0]], dtype=np.float32))

    def test_normalize_roundtrip(self) -> None:
        data = np.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]], dtype=np.float32)
        stats = get_data_stats(data)
        normalized = normalize_data(data, stats)
        recovered = unnormalize_data(normalized, stats)
        np.testing.assert_allclose(recovered, data)


if __name__ == "__main__":
    unittest.main()
