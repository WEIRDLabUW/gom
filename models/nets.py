import math

import flax.linen as nn
import jax
import jax.numpy as jnp


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal embeddings for noise levels."""

    dim: int = 256

    @nn.compact
    def __call__(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        if self.dim % 2 == 1:  # zero pad
            emb = jnp.pad(emb, [[0, 0], [0, 1]])
        return emb


class GaussianFourierEmbedding(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    dim: int = 256
    scale: float = 16
    trainable: bool = True

    @nn.compact
    def __call__(self, x):
        W = self.param(
            "W", jax.nn.initializers.normal(stddev=self.scale), (self.dim // 2,)
        )
        if not self.trainable:
            W = jax.lax.stop_gradient(W)
        x_proj = x[:, None] * W[None, :] * 2 * jnp.pi
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


class LinearBlock(nn.Module):
    """Linear block with layer norm and activation."""

    output_dim: int

    @nn.compact
    def __call__(self, x):
        out = nn.Dense(self.output_dim)(x)
        out = nn.LayerNorm()(out)
        out = nn.swish(out)
        return out


class ConditionalResidualLinearBlock(nn.Module):
    """Conditional residual linear block with FiLM modulation."""

    output_dim: int
    cond_dim: int

    @nn.compact
    def __call__(self, x, cond):
        # First linear layer
        out = LinearBlock(self.output_dim)(x)

        # FiLM modulation
        gamma, beta = jnp.split(nn.Dense(self.output_dim * 2)(cond), 2, axis=-1)
        out = gamma * out + beta

        # Second linear layer
        out = LinearBlock(self.output_dim)(out)

        # Residual connection
        out = out + nn.Dense(self.output_dim)(x)
        return out


class ConditionalUnet1D(nn.Module):
    """Conditional U-Net."""

    output_dim: int
    global_cond_dim: int
    embed_dim: int = 256
    down_dims: tuple = (256, 512, 1024)
    embed_type: str = "positional"

    @nn.compact
    def __call__(self, x, timestep, global_cond):
        assert self.embed_type in ["positional", "fourier"]

        cond_dim = self.embed_dim * 2  # time and conition embeddings
        start_dim = self.down_dims[0]
        mid_dim = self.down_dims[-1]

        # Embed timesteps
        if self.embed_type == "positional":
            t_embed = SinusoidalEmbedding(self.embed_dim)(timestep)
        else:
            t_embed = GaussianFourierEmbedding(self.embed_dim)(timestep)

        # Embed conditions
        hidden_dim = max(self.embed_dim, self.global_cond_dim) * 2
        c_embed = LinearBlock(hidden_dim)(global_cond)
        c_embed = LinearBlock(hidden_dim)(c_embed)
        c_embed = nn.Dense(self.embed_dim)(c_embed)

        # Combine time and condition embeddings
        cond = jnp.concatenate([t_embed, c_embed], axis=-1)

        # Down modules
        h = []
        for out_dim in self.down_dims:
            x = ConditionalResidualLinearBlock(out_dim, cond_dim)(x, cond)
            x = ConditionalResidualLinearBlock(out_dim, cond_dim)(x, cond)
            h.append(x)

        # Mid modules
        x = ConditionalResidualLinearBlock(mid_dim, cond_dim)(x, cond)
        x = ConditionalResidualLinearBlock(mid_dim, cond_dim)(x, cond)

        # Up modules
        for out_dim in reversed(self.down_dims[:-1]):
            x = jnp.concatenate([x, h.pop()], axis=-1)
            x = ConditionalResidualLinearBlock(out_dim, cond_dim)(x, cond)
            x = ConditionalResidualLinearBlock(out_dim, cond_dim)(x, cond)

        x = LinearBlock(start_dim)(x)
        x = nn.Dense(self.output_dim)(x)
        return x
