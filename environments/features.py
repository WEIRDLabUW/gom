from abc import ABC, abstractmethod

import gym
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from torch.utils.data import Dataset
from tqdm.rich import trange

from .datasets import OfflineDataset


class FeatureDataset(Dataset):
    def __init__(self, dataset, env):
        assert isinstance(dataset, OfflineDataset)
        assert isinstance(env, FeatureWrapper)
        self.dataset = dataset

        # Compute features for dataset
        features = []
        next_features = []
        batch_size = 1024
        for i in trange(0, len(self.dataset), batch_size, desc="Featurizing dataset"):
            if i + batch_size > len(self.dataset):
                batch_size = len(self.dataset) - i
            reward = self.dataset.rewards[i : i + batch_size]
            obs = self.dataset.observations[i : i + batch_size]
            next_obs = self.dataset.next_observations[i : i + batch_size]
            features.append(env.feature(obs, reward))
            next_features.append(env.feature(next_obs, reward))
        self.features = np.concatenate(features, axis=0).astype(np.float32)
        self.next_features = np.concatenate(next_features, axis=0).astype(np.float32)

        # Compute statistics for normalization
        self.feature_min = np.min(self.features, axis=0)
        self.feature_max = np.max(self.features, axis=0)

        # Normalize the bias separately
        self.feature_min[-1] = 0
        self.feature_max[-1] = 1

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        data["features"] = self.features[idx]
        data["next_features"] = self.next_features[idx]
        return data


class FeatureWrapper(gym.Wrapper, ABC):
    @property
    @abstractmethod
    def feat_dim(self):
        pass

    @abstractmethod
    def feature(
        self,
        obs: np.ndarray,
        reward: np.ndarray,
    ) -> np.ndarray:
        """
        Compute features for a batch of observations and rewards.
        """
        pass

    def step(self, action):
        obs, original_reward, done, info = self.env.step(action)
        reward = self.feature(obs[None], np.array([original_reward])[None])[0]
        info["original_reward"] = original_reward
        return obs, reward, done, info


class RandMLP(nn.Module):
    out_dim: int
    hidden_dim: int = 32
    normalize: bool = False

    @nn.compact
    def __call__(self, x):
        x = x[:, None, :]
        x = nn.Conv(self.hidden_dim * self.out_dim, (1,))(x)
        x = nn.relu(x)
        x = nn.Conv(
            self.hidden_dim * self.out_dim, (1,), feature_group_count=self.out_dim
        )(x)
        x = nn.relu(x)
        x = nn.Conv(self.out_dim, (1,), feature_group_count=self.out_dim)(x)
        if self.normalize:
            x = nn.tanh(x)
        return x[:, 0, :]


class RandomFeatureWrapper(FeatureWrapper):
    def __init__(self, env, rand_feat_dim, seed=0):
        super().__init__(env)
        self.in_dim = env.observation_space.shape[0]
        self.rand_feat_dim = rand_feat_dim

        self.net = RandMLP(self.rand_feat_dim)
        self.net_params = self.net.init(
            jax.random.PRNGKey(seed), jnp.zeros((1, self.in_dim))
        )
        self.net_params = jax.lax.stop_gradient(self.net_params)

    @property
    def feat_dim(self):
        return self.rand_feat_dim + 1

    def feature(self, obs, reward):
        feat = self.net.apply(self.net_params, obs)
        return np.concatenate([feat, np.ones((len(obs), 1))], axis=-1)


class FourierFeatureWrapper(RandomFeatureWrapper):
    def __init__(self, env, rand_feat_dim, seed=0):
        assert rand_feat_dim % 2 == 0, "Fourier features must have even dimension"
        super().__init__(env, rand_feat_dim // 2, seed)

    @property
    def feat_dim(self):
        return self.rand_feat_dim * 2 + 1

    def feature(self, obs, reward):
        feat = self.net.apply(self.net_params, obs)
        return np.concatenate(
            [np.sin(feat), np.cos(feat), np.ones((len(obs), 1))], axis=-1
        )


class PolynomialFeatureWrapper(FeatureWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.in_dim = env.observation_space.shape[0]

    @property
    def feat_dim(self):
        return int((self.in_dim + 2) * (self.in_dim + 1) / 2)

    def feature(self, obs, reward):
        x = np.ones((len(obs), self.in_dim + 1))
        x[:, :-1] = obs
        x = x[:, None, :] * x[:, :, None]
        triu_inds = np.triu_indices(self.in_dim + 1)
        return x[:, triu_inds[0], triu_inds[1]]


class DummyFeatureWrapper(FeatureWrapper):
    @property
    def feat_dim(self):
        return 1

    def feature(self, obs, reward):
        return reward
