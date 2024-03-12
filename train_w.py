import functools
import os

import hydra
import jax
import jax.numpy as jnp
import wandb
from omegaconf import OmegaConf
from orbax import checkpoint
from torch.utils.data import DataLoader
from tqdm.rich import tqdm

from environments import make_env_and_dataset
from train import build_models, evaluate


@functools.partial(jax.jit, static_argnames=("w_loss_fn"))
def update(w, w_loss_fn, batch):
    # Update w
    w, w_info = w.apply_loss_fn(
        loss_fn=w_loss_fn,
        x=batch["next_features"],
        y=batch["rewards"],
        has_aux=True,
    )

    train_info = {"train/w_loss": w_info["loss"]}
    return w, train_info


@hydra.main(version_base=None, config_path="configs/", config_name="atrl_train_w.yaml")
def train(config):
    # Initialize wandb
    wandb.init(
        project="all-task-rl",
        group=config.env_id,
        job_type=f"{config.algo}_train_w",
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
    )

    # Make environment and dataset
    env, dataset = make_env_and_dataset(
        config.env_id, config.seed, config.feat.type, config.feat.dim
    )

    # Build dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        drop_last=True,
    )

    # Round steps to epochs
    num_epochs = config.training.num_steps // len(dataloader)
    num_steps = num_epochs * len(dataloader)
    OmegaConf.update(config, "training.num_steps", num_steps)

    # Define RNG
    rng = jax.random.PRNGKey(config.seed)

    # Build models
    (
        psi,
        psi_sampler,
        psi_loss_fn,
        policy,
        policy_sampler,
        policy_loss_fn,
        w,
        w_loss_fn,
        planner,
        rng,
    ) = build_models(config, env, dataset, rng)

    # Load psi and policy
    checkpointer = checkpoint.PyTreeCheckpointer()
    options = checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
    checkpoint_manager = checkpoint.CheckpointManager(
        os.path.abspath(config.logdir), checkpointer, options
    )
    target = {"config": config, "psi": psi, "policy": policy}
    step = checkpoint_manager.latest_step()
    ckpt = checkpoint_manager.restore(step, items=target)
    psi, policy = ckpt["psi"], ckpt["policy"]

    # Create new checkpointer
    checkpoint_manager = checkpoint.CheckpointManager(
        os.path.abspath(os.path.join(config.logdir, "w")), checkpointer, options
    )

    # Train w
    step = 0
    pbar = tqdm(total=config.training.num_steps)
    for epoch in range(num_epochs):
        for batch in dataloader:
            batch = {k: jnp.array(v) for k, v in batch.items()}
            w, train_info = update(w, w_loss_fn, batch)
            wandb.log(train_info)

            # Evaluate
            if (step + 1) % config.training.eval_every == 0:
                rng, eval_info = evaluate(
                    config,
                    rng,
                    env,
                    planner,
                    psi,
                    psi_sampler,
                    policy,
                    policy_sampler,
                    w,
                )
                wandb.log(eval_info)

            # Save checkpoint
            if (step + 1) % config.training.save_every == 0:
                ckpt = {"config": config, "psi": psi, "policy": policy, "w": w}
                checkpoint_manager.save(step, ckpt)

            step += 1
            pbar.update(1)

        # Logging
        wandb.log({"train/epoch": epoch})

    pbar.close()


if __name__ == "__main__":
    train()
