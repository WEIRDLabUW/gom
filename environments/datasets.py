import glob
import os
import sys

import numpy as np
import torch
from torch.multiprocessing import Pool
from torch.utils.data import Dataset
from tqdm.rich import trange


class OfflineDataset(Dataset):
    def __init__(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        dones,
    ):
        if len(observations.shape) == 4:
            obs_dtype = np.uint8
        else:
            obs_dtype = np.float32
        self.observations = np.array(observations).astype(obs_dtype)
        self.actions = np.array(actions).astype(np.float32)
        self.rewards = np.array(rewards).astype(np.float32).reshape(-1, 1)
        self.next_observations = np.array(next_observations).astype(obs_dtype)
        self.dones = np.array(dones).astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return dict(
            observations=self.observations[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_observations=self.next_observations[idx],
            dones=self.dones[idx],
        )


class D4RLDataset(OfflineDataset):
    def __init__(self, env):
        import d4rl

        dataset = d4rl.qlearning_dataset(env)

        observations = dataset["observations"]
        actions = dataset["actions"]
        next_observations = dataset["next_observations"]
        rewards = dataset["rewards"]
        dones = dataset["terminals"]

        if "antmaze" in env.spec.id:
            # Compute dense rewards
            goal = np.array(env.target_goal)
            dists_to_goal = np.linalg.norm(
                goal[None] - next_observations[:, :2], axis=-1
            )
            rewards = np.exp(-dists_to_goal / 20)
        elif "kitchen" in env.spec.id:
            # Remove goals from observations
            observations = observations[:, :30]
            next_observations = observations[:, :30]

        super().__init__(observations, actions, rewards, next_observations, dones)


class RoboverseDataset(OfflineDataset):
    def __init__(self, env, task, data_dir="data/roboverse"):
        if task == "pickplace-v0":
            prior_data_path = os.path.join(data_dir, "pickplace_prior.npy")
            task_data_path = os.path.join(data_dir, "pickplace_task.npy")
        elif task == "doubledraweropen-v0":
            prior_data_path = os.path.join(data_dir, "closed_drawer_prior.npy")
            task_data_path = os.path.join(data_dir, "drawer_task.npy")
        elif task == "doubledrawercloseopen-v0":
            prior_data_path = os.path.join(data_dir, "blocked_drawer_1_prior.npy")
            task_data_path = os.path.join(data_dir, "drawer_task.npy")
        else:
            raise NotImplementedError("Unsupported roboverse task")

        prior_data = np.load(prior_data_path, allow_pickle=True)
        task_data = np.load(task_data_path, allow_pickle=True)

        full_data = np.concatenate((prior_data, task_data))
        dict_data = {}
        for key in [
            "observations",
            "actions",
            "rewards",
            "next_observations",
            "terminals",
        ]:
            full_values = []
            for traj in full_data:
                values = traj[key]
                if key == "observations" or key == "next_observations":
                    full_values += [env.observation(obs) for obs in values]
                else:
                    full_values += values
            dict_data[key] = np.array(full_values)

        super().__init__(
            dict_data["observations"],
            dict_data["actions"],
            dict_data["rewards"],
            dict_data["next_observations"],
            dict_data["terminals"],
        )


class AntMazePreferenceDataset(OfflineDataset):
    def __init__(self, env):
        import d4rl

        dataset = d4rl.qlearning_dataset(
            env,
            h5path="data/d4rl/Ant_maze_obstacle_noisy_multistart_True_multigoal_True.hdf5",
        )
        rewards = env.compute_reward(dataset["next_observations"])
        super().__init__(
            dataset["observations"],
            dataset["actions"],
            rewards,
            dataset["next_observations"],
            dataset["terminals"],
        )
