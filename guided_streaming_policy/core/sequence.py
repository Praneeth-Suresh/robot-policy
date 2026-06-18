from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class SequencePointer:
    """
    Container for sample extraction pointers.
    """

    buffer_start_idx: int
    buffer_end_idx: int
    sample_start_idx: int
    sample_end_idx: int

    def __iter__(self):
        return iter(
            [
                self.buffer_start_idx,
                self.buffer_end_idx,
                self.sample_start_idx,
                self.sample_end_idx,
            ]
        )


def create_sequence_pointers(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
) -> List[SequencePointer]:
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append(
                SequencePointer(
                    buffer_start_idx,
                    buffer_end_idx,
                    sample_start_idx,
                    sample_end_idx,
                )
            )
    return indices


def extract_sequence(train_data, sequence_length, ptr: SequencePointer):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[ptr.buffer_start_idx : ptr.buffer_end_idx]
        data = sample
        if (ptr.sample_start_idx > 0) or (ptr.sample_end_idx < sequence_length):
            data = np.zeros(shape=(sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype)
            if ptr.sample_start_idx > 0:
                data[: ptr.sample_start_idx] = sample[0]
            if ptr.sample_end_idx < sequence_length:
                data[ptr.sample_end_idx :] = sample[-1]
            data[ptr.sample_start_idx : ptr.sample_end_idx] = sample
        result[key] = data
    return result


def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    stats = {
        "min": np.min(data, axis=0),
        "max": np.max(data, axis=0),
    }
    return stats


def normalize_data(data, stats):
    ndata = (data - stats["min"]) / (stats["max"] - stats["min"])
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats["max"] - stats["min"]) + stats["min"]
    return data
