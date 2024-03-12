import os

import hydra
import imageio
import jax
import numpy as np
from orbax import checkpoint

from environments import make_env_and_dataset
from train import build_models, evaluate


@hydra.main(version_base=None, config_path="configs/", config_name="atrl.yaml")
def main(config):
    # Make environment and dataset
    env, dataset = make_env_and_dataset(
        config.env_id, config.seed, config.feat.type, config.feat.dim
    )

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

    # Load checkpoints
    checkpointer = checkpoint.PyTreeCheckpointer()
    options = checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
    if os.path.exists(os.path.join(config.logdir, "w")):
        checkpoint_manager = checkpoint.CheckpointManager(
            os.path.abspath(os.path.join(config.logdir, "w")), checkpointer, options
        )
    else:
        checkpoint_manager = checkpoint.CheckpointManager(
            os.path.abspath(config.logdir), checkpointer, options
        )
    target = {"config": config, "psi": psi, "policy": policy, "w": w}
    step = checkpoint_manager.latest_step()
    ckpt = checkpoint_manager.restore(step, items=target)
    psi, policy, w = ckpt["psi"], ckpt["policy"], ckpt["w"]

    # Evaluate online
    ep_returns = []
    ep_successes = []
    for i in range(config.eval.num_episodes):
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
        ep_returns.append(eval_info["test/return"])
        ep_successes.append(float(eval_info["test/return"] > 0))

        print(f"Episode {i}")
        print("Return:", eval_info["test/return"])
        print("Success:", float(eval_info["test/return"] > 0))
        imageio.mimsave(
            os.path.join(
                config.logdir,
                f"{config.planning.planner}_{config.planning.guidance_coef:.2f}_obs_{i}.gif",
            ),
            eval_info["test/video"].data[0].transpose(0, 2, 3, 1),
            fps=30,
        )
        if "test/psi_video" in eval_info:
            imageio.mimsave(
                os.path.join(
                    config.logdir,
                    f"{config.planning.planner}_{config.planning.guidance_coef:.2f}_psi_{i}.gif",
                ),
                eval_info["test/psi_video"].data[0].transpose(0, 2, 3, 1),
                fps=30,
            )

    print(f"Average return: {np.mean(ep_returns):.4f} +- {np.std(ep_returns):.4f}")
    print(f"Average success: {np.mean(ep_successes):.4f} +- {np.std(ep_successes):.4f}")


if __name__ == "__main__":
    main()
