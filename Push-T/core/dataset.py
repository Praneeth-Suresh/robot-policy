from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import zarr

@dataclass
class SequencePointer:
    """
    Container for sample extraction pointers.
    """

    """Starting index in the original data buffer to begin extraction."""
    buffer_start_idx: int

    """Ending index in the original data buffer (exclusive) for extraction."""
    buffer_end_idx: int
    
    """Starting index within the padded sample."""
    sample_start_idx: int
    
    """Ending index within the padded sample."""
    sample_end_idx: int

    def __iter__(self):
        """Allow unpacking: buffer_start, buffer_end, sample_start, sample_end = indices"""
        return iter([self.buffer_start_idx, self.buffer_end_idx, 
                     self.sample_start_idx, self.sample_end_idx])

def create_sequence_pointers(
        episode_ends: np.ndarray, sequence_length: int,
        pad_before: int=0, pad_after: int=0,
    ) -> List[SequencePointer]:
    """
    Create sample indices for extracting sequences from episode data with padding support.
    
    This function generates indices for extracting fixed-length sequences from a dataset
    composed of multiple episodes. It handles padding at the beginning and end of episodes
    to ensure all timesteps can be used as starting points for sequences, even near episode
    boundaries.
    
    Parameters
    ----------
    episode_ends : (np.ndarray, shape=(E,))
        Array containing the end indices (one-past-the-last) for each episode.
        For example, if episodes have lengths [10, 15, 8], then episode_ends = [10, 25, 33].
    sequence_length : int
        Length of the sequences to extract from the dataset.
    pad_before : int, default=0
        Number of timesteps to pad before the episode start. This allows sequences
        to start before the actual episode data by repeating the first values of all data keys.
    pad_after : int, default=0
        Number of timesteps to pad after the episode end. This allows sequences
        to extend beyond the actual episode data by repeating the last values of all data keys.
    
    Returns
    -------
    List[SequencePointer]
        List of SequencePointer objects for extracting sequences from all episodes.
        Can be converted to numpy array using np.array(indices) for compatibility.
        
    Notes
    -----
    The function handles episode boundaries by:
    1. Computing valid start positions for sequences within each episode.
    2. Accounting for padding requirements at episode boundaries.
    3. Returning indices that can be used to extract properly padded sequences.
    
    The returned indices are designed to work with a sampling function that can
    handle the padding by repeating boundary values when accessing out-of-bounds indices.
    
    Example
    -------
    >>> episode_ends = np.array([10, 25, 33])  # 3 episodes of lengths 10, 15, 8
    >>> indices = create_sample_indices(episode_ends, sequence_length=5, pad_before=2, pad_after=1)
    >>> # Returns list of SequencePointer objects for all possible 5-step sequences
    """
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append(SequencePointer(
                buffer_start_idx,
                buffer_end_idx,
                sample_start_idx,
                sample_end_idx,
            ))
    return indices

def extract_sequence(train_data, sequence_length, ptr: SequencePointer):
    """
    Extract and pad a sequence from training data using specified indices.
    
    This function extracts a sequence from the training data buffer and applies padding
    if necessary to ensure the output has the exact sequence_length. Padding is applied
    by repeating boundary values when the sequence extends beyond the available data.
    
    Parameters
    ----------
    train_data : dict
        Dictionary containing training data arrays. Each key maps to a numpy array
        of shape (N, feature_dim) where N is the total number of timesteps.
        Common keys include 'action' and 'obs' (observations).
    sequence_length : int
        Desired length of the output sequence. All returned arrays will have
        this length in the first dimension.
    ptr : SequencePointer
        Container of sequence pointer indices for extracting and placing the
        sequence data.
    
    Returns
    -------
    dict
        Dictionary with the same keys as train_data, where each value is a numpy array
        of shape (sequence_length, feature_dim). The arrays contain the extracted data
        with appropriate padding applied.
        
    Notes
    -----
    The function handles three cases:
    1. No padding needed: When sample_start_idx=0 and sample_end_idx=sequence_length,
       the extracted data is returned as-is.
    2. Start padding: When sample_start_idx > 0, the first sample_start_idx positions
       are filled with the first value of the extracted sample.
    3. End padding: When sample_end_idx < sequence_length, positions from sample_end_idx
       to sequence_length are filled with the last value of the extracted sample.
    
    This function is typically used with indices generated by create_sample_indices()
    to handle episode boundaries and ensure consistent sequence lengths for training.
    
    Example
    -------
    >>> train_data = {'action': np.array([[1], [2], [3], [4], [5]]), 
    ...               'obs': np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])}
    >>> indices = SampleIndices(buffer_start_idx=1, buffer_end_idx=4,
    ...                         sample_start_idx=2, sample_end_idx=5)
    >>> result = sample_sequence(train_data, sequence_length=6, indices)
    >>> # Extracts [2,3,4] from buffer indices 1:4, but places them at positions 2:5
    >>> # result['action'] = [[2], [2], [2], [3], [4], [4]]  # padded at start and end
    >>> # result['obs'] = [[0.2], [0.2], [0.2], [0.3], [0.4], [0.4]]  # same padding pattern
    """
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[ptr.buffer_start_idx:ptr.buffer_end_idx]
        data = sample
        if (ptr.sample_start_idx > 0) or (ptr.sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if ptr.sample_start_idx > 0:
                data[:ptr.sample_start_idx] = sample[0]
            if ptr.sample_end_idx < sequence_length:
                data[ptr.sample_end_idx:] = sample[-1]
            data[ptr.sample_start_idx:ptr.sample_end_idx] = sample
        result[key] = data
    return result

def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])  # to [0, 1]
    ndata = ndata * 2 - 1  # to [-1, 1]
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2  # to [0, 1]
    data = ndata * (stats['max'] - stats['min']) + stats['min']  # to original
    return data

class PushTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path,
                 pred_horizon, obs_horizon, action_horizon):

        # Read from zarr dataset
        dataset_root = zarr.open(dataset_path, 'r')
        
        # All demonstration episodes are concatenated in the first dimension N
        train_data = {
            'action': dataset_root['data']['action'][:],  # (N, action_dim)
            'obs': dataset_root['data']['state'][:],  # (N, obs_dim)
        }
        # Marks one-past the last index for each episode
        episode_ends = dataset_root['meta']['episode_ends'][:]

        # |o|o|                             observations: 2
        # | |a|a|a|a|a|a|a|a|               actions executed: 8
        # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

        # Compute start and end of each state-action sequence,
        # also add padding such that each timestep in the dataset is seen
        self.sequence_pointers = create_sequence_pointers(
            episode_ends = episode_ends,
            sequence_length = pred_horizon,
            pad_before = obs_horizon-1,
            pad_after = action_horizon-1,
        )

        # Compute statistics and normalized data to [-1, 1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        """Count of all possible segments of the dataset"""
        return len(self.sequence_pointers)

    def __getitem__(self, idx):
        # Get the sequence pointer for this datapoint
        ptr = self.sequence_pointers[idx]

        # Get normalized data using these indices
        nsample = extract_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            ptr=ptr,
        )

        # discard unused observations
        nsample['obs'] = nsample['obs'][:self.obs_horizon,:]
        return nsample
