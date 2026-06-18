import torch
import zarr

from guided_streaming_policy.core.sequence import (
    SequencePointer,
    create_sequence_pointers,
    extract_sequence,
    get_data_stats,
    normalize_data,
    unnormalize_data,
)


class PushTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, pred_horizon, obs_horizon, action_horizon):
        dataset_root = zarr.open(dataset_path, "r")

        train_data = {
            "action": dataset_root["data"]["action"][:],
            "obs": dataset_root["data"]["state"][:],
        }
        episode_ends = dataset_root["meta"]["episode_ends"][:]

        self.sequence_pointers = create_sequence_pointers(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

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
        return len(self.sequence_pointers)

    def __getitem__(self, idx):
        ptr = self.sequence_pointers[idx]

        nsample = extract_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            ptr=ptr,
        )

        nsample["obs"] = nsample["obs"][: self.obs_horizon, :]
        return nsample


__all__ = [
    "PushTDataset",
    "SequencePointer",
    "create_sequence_pointers",
    "extract_sequence",
    "get_data_stats",
    "normalize_data",
    "unnormalize_data",
]
