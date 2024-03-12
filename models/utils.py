from typing import Any, Callable, Optional, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
import optax
import functools

from models import sde_lib
from utils import batch_mul

# Define types
Params = flax.core.FrozenDict[str, Any]
ModuleMethod = Union[str, Callable, None]
nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


class TrainState(flax.struct.PyTreeNode):
    """
    Core abstraction of a model in this repository. Fully backward compatible with standard flax.training.TrainState.

    Creation:
    ```
        model_def = nn.Dense(12) # or any other flax.linen Module
        _, params = model_def.init(jax.random.PRNGKey(0), jnp.ones((1, 4))).pop('params')
        model = TrainState.create(model_def, params, tx=None) # Optionally, pass in an optax optimizer
    ```

    Usage:
    ```
        y = model(jnp.ones((1, 4))) # By default, uses the `__call__` method of the model_def and params stored in TrainState
        y = model(jnp.ones((1, 4)), params=params) # You can pass in params (useful for gradient computation)
        y = model(jnp.ones((1, 4)), method=method) # You can apply a different method as well
    ```

    More complete example:
    ```
        def loss(params):
            y_pred = model(x, params=params)
            return jnp.mean((y - y_pred) ** 2)

        grads = jax.grad(loss)(model.params)
        new_model = model.apply_gradients(grads=grads) # Alternatively, new_model = model.apply_loss_fn(loss_fn=loss)
    ```
    """

    step: int
    apply_fn: Callable = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Params
    tx: Optional[optax.GradientTransformation] = nonpytree_field()
    opt_state: Optional[optax.OptState]

    @classmethod
    def create(
        cls,
        model_def: nn.Module,
        params: Params,
        tx: Optional[optax.GradientTransformation] = None,
        **kwargs,
    ) -> "TrainState":
        # Initialize optimizer state
        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def __call__(
        self,
        *args,
        params: Params = None,
        method: ModuleMethod = None,
        **kwargs,
    ):
        """
        Internally calls model_def.apply_fn with the following logic:

        Arguments:
            params: If not None, use these params instead of the ones stored in the model.
            extra_variables: Additional variables to pass into apply_fn (overrides model.extra_variables if they exist)
            method: If None, use the `__call__` method of the model_def. If a string, uses
                the method of the model_def with that name (e.g. 'encode' -> model_def.encode).
                If a function, uses that function.

        """
        if params is None:
            params = self.params

        variables = {"params": params}

        if isinstance(method, str):
            method = getattr(self.model_def, method)

        return self.apply_fn(variables, *args, method=method, **kwargs)

    def apply_gradients(self, *, grads, **kwargs):
        """
        Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

        Note that internally this function calls `.tx.update()` followed by a call
        to `optax.apply_updates()` to update `params` and `opt_state`.

        Args:
            grads: Gradients that have the same pytree structure as `.params`.
            **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

        Returns:
            An updated instance of `self` with `step` incremented by one, `params`
            and `opt_state` updated by applying `grads`, and additional attributes
            replaced as specified by `kwargs`.
        """
        # Update params and opt_state
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    def apply_loss_fn(self, *, loss_fn, pmap_axis=None, has_aux=False, **kwargs):
        """
        Takes a gradient step towards minimizing `loss_fn`. Internally, this calls
        `jax.grad` followed by `TrainState.apply_gradients`. If pmap_axis is provided,
        additionally it averages gradients (and info) across devices before performing update.
        """
        if has_aux:
            grads, info = jax.grad(loss_fn, has_aux=has_aux)(self.params, **kwargs)
            if pmap_axis is not None:
                grads = jax.lax.pmean(grads, axis_name=pmap_axis)
                info = jax.lax.pmean(info, axis_name=pmap_axis)
            return self.apply_gradients(grads=grads), info

        else:
            grads = jax.grad(loss_fn, has_aux=has_aux)(self.params, **kwargs)
            if pmap_axis is not None:
                grads = jax.lax.pmean(grads, axis_name=pmap_axis)
            return self.apply_gradients(grads=grads)

    def __getattr__(self, name):
        """
        Syntatic sugar for calling methods of the model_def directly.

        Example:
        ```
            model(x, method='encode')
            model.encode(x) # Same as last
        """
        method = getattr(self.model_def, name)
        return functools.partial(self.__call__, method=method)


class EMATrainState(TrainState):
    """
    TrainState with EMA parameters. Fully backward compatible with standard flax.training.TrainState.
    """

    ema_params: Params
    ema_rate: float

    @classmethod
    def create(
        cls,
        model_def: nn.Module,
        params: Params,
        ema_rate: float = 0.999,
        tx: Optional[optax.GradientTransformation] = None,
        **kwargs,
    ) -> "TrainState":
        # Initialize ema_params as params
        ema_params = params

        # Initialize optimizer state
        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            ema_params=ema_params,
            ema_rate=ema_rate,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def apply_gradients(self, *, grads, **kwargs):
        """
        In addition to the original apply_gradients functionalities, also update EMA parameters.
        """
        # Update params and opt_state
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        # Update EMA params
        new_ema_params = jax.tree_map(
            lambda ema_p, p: ema_p * self.ema_rate + p * (1.0 - self.ema_rate),
            self.ema_params,
            new_params,
        )

        return self.replace(
            step=self.step + 1,
            params=new_params,
            ema_params=new_ema_params,
            opt_state=new_opt_state,
            **kwargs,
        )


def get_score_fn(
    sde: sde_lib.SDE,
    model_def: nn.Module,
    params: Params = None,
    continuous: bool = True,
):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model_def: An `nn.Module` object that represents the architecture of the score-based model.
      params: A dictionary that contains trainable parameters of the score-based model.
      continuous: If `True`, the score-based model is expected to directly take continuous time steps.

    Returns:
      A score function.
    """

    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):

        def score_fn(x, t, cond, rng=None):
            # Scale neural network output by standard deviation and flip sign
            if continuous or isinstance(sde, sde_lib.subVPSDE):
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                label = t * 999
                out = model_def.apply({"params": params}, x, label, cond)
                std = sde.marginal_prob(jnp.zeros_like(x), t)[1]
            else:
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # Use continuous time for model input and discrete time for std
                label = t * sde.N / sde.T
                out = model_def.apply({"params": params}, x, label, cond)
                std = sde.sqrt_1m_alphas_cumprod[label.astype(jnp.int32)]

            score = batch_mul(-out, 1.0 / std)
            return score

    elif isinstance(sde, sde_lib.VESDE):

        def score_fn(x, t, cond, rng=None):
            if sde.linear is False:
                if continuous:
                    label = sde.marginal_prob(jnp.zeros_like(x), t)[1]
                else:
                    # For VE-trained models, t=0 corresponds to the highest noise level
                    label = ((sde.T - t) * sde.N).round().astype(jnp.int32)
                score = model_def.apply({"params": params}, x, label, cond)
            else:
                assert continuous
                label = t * 999
                out = model_def.apply({"params": params}, x, label, cond)
                std = sde.marginal_prob(jnp.zeros_like(x), t)[1]
                score = batch_mul(-out, 1.0 / std)

            return score

    else:
        raise NotImplementedError(f"{sde.__class__.__name__} not yet supported.")

    return score_fn


def get_loss_fn(
    sde: sde_lib.SDE,
    model_def: nn.Module,
    scaler: Callable,
    continuous: bool = True,
    likelihood_weighting: bool = False,
    importance_weighting: bool = False,
    eps: float = 0.0,
):
    """Create a loss function for training with arbirary SDEs.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model_def: An `nn.Module` object that represents the architecture of the score-based model.
      scaler: A function that normalizes the data.
      continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
        ad-hoc interpolation to take continuous time steps.
      likelihood_weighting: If `True`, weight the mixture of score matching losses
        according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
      importance_weighting: If `True`, use importance weighting to reduce the variance of likelihood weighting.
      eps: A `float` number. The smallest time step to sample from.

    Returns:
      A loss function.
    """

    def loss_fn(params, rng, x, cond):
        """Compute the loss function.

        Args:
          params: A dictionary that contains trainable parameters of the score-based model.
          rng: A JAX random state.
          x: A mini-batch of inputs.
          cond: A mini-batch of conditions.

        Returns:
          loss: A scalar that represents the average loss value across the mini-batch.
        """

        # Normalize data
        x = scaler(x)

        # Get score function
        score_fn = get_score_fn(sde, model_def, params, continuous)

        # Sample timesteps
        rng, step_rng = random.split(rng)
        if likelihood_weighting and importance_weighting:
            t = sde.sample_importance_weighted_time_for_likelihood(
                step_rng, (x.shape[0],), eps=eps
            )
        else:
            t = random.uniform(step_rng, (x.shape[0],), minval=eps, maxval=sde.T)

        # Sample noise
        rng, step_rng = random.split(rng)
        z = random.normal(step_rng, x.shape)

        # Add noise to data
        if not continuous and isinstance(sde, sde_lib.VPSDE):
            label = (t * sde.N / sde.T).astype(jnp.int32)
            mean = batch_mul(sde.sqrt_alphas_cumprod[label], x)
            std = sde.sqrt_1m_alphas_cumprod[label]
        else:
            mean, std = sde.marginal_prob(x, t)
        x_t = mean + batch_mul(std, z)

        # Predict score from noisy data
        score = score_fn(x_t, t, cond)

        # Compute loss
        if likelihood_weighting and not importance_weighting:
            g2 = sde.sde(jnp.zeros_like(x), t)[1] ** 2
            losses = jnp.square(score + batch_mul(z, 1.0 / std))
            losses = jnp.mean(losses.reshape((losses.shape[0], -1)), axis=-1) * g2
        else:
            losses = jnp.square(batch_mul(score, std) + z)
            losses = jnp.mean(losses.reshape((losses.shape[0], -1)), axis=-1)

        loss = jnp.mean(losses)
        return loss, {"loss": loss}

    return loss_fn
