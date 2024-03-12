"""Various sampling methods."""

import abc
import functools

import jax
import jax.numpy as jnp
import jax.random as random

from models import sde_lib
from models import utils as mutils

_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


def get_corrector(name):
    return _CORRECTORS[name]


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, rng, x, t, cond, g=None):
        """One update of the predictor.

        Args:
          rng: A JAX random state.
          x: A JAX array representing the current state.
          t: A JAX array representing the current time step.
          cond: A JAX array representing the global condition.
          g: A JAX array representing diffusion guidance.

        Returns:
          x: A JAX array of the next state.
          x_mean: A JAX array. The next state without random noise. Useful for denoising.
        """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, rng, x, t, cond, g=None):
        """One update of the corrector.

        Args:
          rng: A JAX random state.
          x: A JAX array representing the current state.
          t: A JAX array representing the current time step.
          cond: A JAX array representing the global condition.
          g: A JAX array representing diffusion guidance.

        Returns:
          x: A JAX array of the next state.
          x_mean: A JAX array. The next state without random noise. Useful for denoising.
        """
        pass


@register_predictor(name="ddpm")
class DDPMPredictor(Predictor):
    """The DDPM predictor. Currently only supports VP SDEs."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)
        if not isinstance(sde, sde_lib.VPSDE):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )
        assert not probability_flow, "Probability flow not supported by DDPM sampling"

    def update_fn(self, rng, x, t, cond, g=None):
        # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm.py

        # Get previous timesteps
        sde = self.sde
        timesteps = (t * sde.N / sde.T).astype(jnp.int32)
        prev_timesteps = timesteps - jnp.ones_like(timesteps)

        # Compute alphas, betas
        alpha_prod_t = sde.alphas_cumprod[timesteps, None]
        alpha_prod_t_prev = jnp.where(
            prev_timesteps[:, None] >= 0,
            sde.alphas_cumprod[prev_timesteps.clip(0), None],
            jnp.ones_like(alpha_prod_t),
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # Compute score
        score = self.score_fn(x, t, cond)
        # Add diffusion guidance
        if g is not None:
            score += g
        # Score is epsilon negated and scaled by 1 / sqrt(beta_prod_t)
        eps = -jnp.sqrt(beta_prod_t) * score

        # Predict original x
        pred_orig_x = (x - jnp.sqrt(beta_prod_t) * eps) / jnp.sqrt(alpha_prod_t)
        pred_orig_x = pred_orig_x.clip(-1, 1)

        # Predict previous x
        pred_orig_x_coeff = jnp.sqrt(alpha_prod_t_prev) * current_beta_t / beta_prod_t
        current_x_coeff = jnp.sqrt(current_alpha_t) * beta_prod_t_prev / beta_prod_t
        pred_prev_x = pred_orig_x_coeff * pred_orig_x + current_x_coeff * x

        # Add noise
        noise = random.normal(rng, x.shape)
        variance = beta_prod_t_prev / beta_prod_t * current_beta_t
        x = pred_prev_x + jnp.sqrt(variance) * noise
        return x, pred_prev_x


@register_predictor(name="ddim")
class DDIMPredictor(Predictor):
    """The DDIM predictor. Currently only supports VP SDEs."""

    def __init__(
        self, sde, score_fn, probability_flow=False, n_inference_steps=None, eta=0.0
    ):
        super().__init__(sde, score_fn, probability_flow)
        if not isinstance(sde, sde_lib.VPSDE):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )
        assert not probability_flow, "Probability flow not supported by DDIM sampling"

        if n_inference_steps is None:
            n_inference_steps = sde.N
        self.interval = sde.N // n_inference_steps
        self.eta = eta

    def update_fn(self, rng, x, t, cond, g=None):
        # https://github.com/huggingface/diffusers/blob/v0.24.0/src/diffusers/schedulers/scheduling_ddim.py

        # Get previous timesteps
        sde = self.sde
        timesteps = (t * sde.N / sde.T).astype(jnp.int32)
        prev_timesteps = timesteps - jnp.ones_like(timesteps) * self.interval

        # Compute alphas, betas
        alpha_prod_t = sde.alphas_cumprod[timesteps, None]
        alpha_prod_t_prev = jnp.where(
            prev_timesteps[:, None] >= 0,
            sde.alphas_cumprod[prev_timesteps.clip(0), None],
            jnp.ones_like(alpha_prod_t),
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # Compute score
        score = self.score_fn(x, t, cond)
        # Add diffusion guidance
        if g is not None:
            score += g
        # Score is epsilon negated and scaled by 1 / sqrt(beta_prod_t)
        eps = -jnp.sqrt(beta_prod_t) * score

        # Predict original x
        pred_orig_x = (x - jnp.sqrt(beta_prod_t) * eps) / jnp.sqrt(alpha_prod_t)
        pred_orig_x = pred_orig_x.clip(-1, 1)

        # Rederive epsilon from clipped original x
        if g is not None:
            eps = (x - jnp.sqrt(alpha_prod_t) * pred_orig_x) / jnp.sqrt(beta_prod_t)

        # Compute variance
        variance = beta_prod_t_prev / beta_prod_t * current_beta_t
        std_dev_t = self.eta * jnp.sqrt(variance)

        # Compute direction pointing to x_t
        pred_x_direction = jnp.sqrt(1 - alpha_prod_t_prev - std_dev_t**2) * eps

        # Predict previous x
        pred_prev_x = jnp.sqrt(alpha_prod_t_prev) * pred_orig_x + pred_x_direction

        # Add noise
        noise = random.normal(rng, x.shape)
        x = pred_prev_x + std_dev_t * noise
        return x, pred_prev_x


@register_predictor(name="none")
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde, score_fn, probability_flow=False):
        pass

    def update_fn(self, rng, x, t, cond, g=None):
        return x, x


@register_corrector(name="none")
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, score_fn, snr, n_steps):
        pass

    def update_fn(self, rng, x, t, cond, g=None):
        return x, x


def shared_predictor_update_fn(
    params,
    rng,
    x,
    t,
    cond,
    g,
    sde,
    model_def,
    predictor,
    probability_flow,
    continuous,
    n_inference_steps,
    eta,
):
    """A wrapper that configures and returns the update function of predictors."""
    score_fn = mutils.get_score_fn(sde, model_def, params, continuous)
    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(sde, score_fn, probability_flow)
    elif predictor == DDIMPredictor:
        predictor_obj = predictor(
            sde, score_fn, probability_flow, n_inference_steps, eta
        )
    else:
        predictor_obj = predictor(sde, score_fn, probability_flow)
    return predictor_obj.update_fn(rng, x, t, cond, g)


def shared_corrector_update_fn(
    params,
    rng,
    x,
    t,
    cond,
    g,
    sde,
    model_def,
    corrector,
    continuous,
    snr,
    n_steps,
):
    """A wrapper tha configures and returns the update function of correctors."""
    score_fn = mutils.get_score_fn(sde, model_def, params, continuous)
    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
    else:
        corrector_obj = corrector(sde, score_fn, snr, n_steps)
    return corrector_obj.update_fn(rng, x, t, cond, g)


def get_pc_sampler(
    sde,
    model_def,
    shape,
    predictor,
    corrector,
    inverse_scaler,
    continuous=False,
    n_inference_steps=None,
    denoise=False,
    probability_flow=False,
    n_corrector_steps=1,
    snr=0.16,
    eps=0.0,
    eta=0.0,
):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      model_def: An `nn.Module` object that represents the architecture of a time-dependent score-based model.
      shape: A tuple of integers representing the shape of a single sample.
      predictor: A `str` representing the predictor algorithm.
      corrector: A `str` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      continuous: `True` indicates that the score model was continuously trained.
      n_inference_steps: An integer. The number of steps to run the sampling process.
      denoise: If `True`, add one-step denoising to the final samples.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      n_corrector_steps: An integer. The number of corrector steps per predictor update.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      eta: A `float` number. The variance coefficient of DDIM predictor.

    Returns:
      A sampling function that takes random states, and a replcated training state and returns samples as well as
      the number of function evaluations during sampling.
    """

    if n_inference_steps is None:
        n_inference_steps = sde.N
    timesteps = jnp.linspace(sde.T, eps, n_inference_steps, endpoint=False)

    # Create predictor & corrector update functions
    predictor = get_predictor(predictor.lower())
    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        sde=sde,
        model_def=model_def,
        predictor=predictor,
        probability_flow=probability_flow,
        continuous=continuous,
        n_inference_steps=n_inference_steps,
        eta=eta,
    )

    corrector = get_corrector(corrector.lower())
    corrector_update_fn = functools.partial(
        shared_corrector_update_fn,
        sde=sde,
        model_def=model_def,
        corrector=corrector,
        continuous=continuous,
        snr=snr,
        n_steps=n_corrector_steps,
    )

    def pc_sampler(params, rng, cond, g=None):
        """The PC sampler function.

        Args:
          params: A `flax.core.FrozenDict` representing the parameters of the score-based model.
          rng: A JAX random state.
          cond: A JAX array representing the global condition.
          g: A JAX array representing diffusion guidance.
        Returns:
          Samples, number of function evaluations
        """
        # Initial sample
        batch_shape = (cond.shape[0],) + shape
        rng, step_rng = random.split(rng)
        x = sde.prior_sampling(step_rng, batch_shape)

        def loop_body(i, val):
            rng, x, x_mean = val
            t = jnp.ones(batch_shape[0]) * timesteps[i]
            rng, step_rng = random.split(rng)
            x, x_mean = corrector_update_fn(params, step_rng, x, t, cond, g)
            rng, step_rng = random.split(rng)
            x, x_mean = predictor_update_fn(params, step_rng, x, t, cond, g)
            return rng, x, x_mean

        _, x, x_mean = jax.lax.fori_loop(0, n_inference_steps, loop_body, (rng, x, x))
        # Denoising is equivalent to running one predictor step without adding noise.
        return inverse_scaler(x_mean if denoise else x), n_inference_steps * (
            n_corrector_steps + 1
        )

    return pc_sampler
