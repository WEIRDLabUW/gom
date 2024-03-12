from functools import partial

import gym

gym.logger.set_level(40)

from .features import (
    FeatureDataset,
    RandomFeatureWrapper,
    FourierFeatureWrapper,
    PolynomialFeatureWrapper,
    DummyFeatureWrapper,
)
from .wrappers import CastObs, TimeLimit


def make_env_and_dataset(
    env_id,
    seed,
    feature=None,
    feature_dim=256,
):
    suite, task = env_id.split("-", 1)
    if suite in ["maze2d", "antmaze", "kitchen"]:
        import d4rl
        from .datasets import D4RLDataset
        from .wrappers import AntMazeWrapper, KitchenWrapper

        env = gym.make(env_id)
        if suite == "antmaze":
            env = AntMazeWrapper(env)
        elif suite == "kitchen":
            env = KitchenWrapper(env)
        dataset = D4RLDataset(env)
    elif suite == "roboverse":
        import roboverse
        from .datasets import RoboverseDataset
        from .wrappers import RoboverseWrapper

        if task == "pickplace-v0":
            taskname = "Widow250PickTray-v0"
            horizon = 40
        elif task == "doubledraweropen-v0":
            taskname = "Widow250DoubleDrawerOpenGraspNeutral-v0"
            horizon = 50
        elif task == "doubledrawercloseopen-v0":
            taskname = "Widow250DoubleDrawerCloseOpenGraspNeutral-v0"
            horizon = 80
        else:
            raise NotImplementedError("Unsupported roboverse task")
        env = roboverse.make(taskname, observation_img_dim=128, transpose_image=False)
        env = RoboverseWrapper(env)
        env = TimeLimit(env, horizon)
        dataset = RoboverseDataset(env, task)
    elif suite == "multimodal":
        task, mode = task.split("-")
        mode = int(mode)
        assert mode in [0, 1]
        import d4rl
        from .wrappers import AntMazePreferenceWrapper
        from .datasets import AntMazePreferenceDataset

        env = gym.make("antmaze-obstacle-v2")
        env = AntMazePreferenceWrapper(env, mode)
        dataset = AntMazePreferenceDataset(env)
    elif suite == "multigoal":
        import d4rl
        from .wrappers import AntMazeMultigoalWrapper
        from .datasets import AntMazePreferenceDataset

        task, mode = task.split("-")
        mode = int(mode)
        env = gym.make("antmaze-medium-diverse-v2")
        env = AntMazeMultigoalWrapper(env, mode)
        dataset = AntMazePreferenceDataset(env)
    else:
        raise NotImplementedError

    # Cast observation dtype to float32
    env = CastObs(env)

    # Set seed
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    if feature is not None:
        if feature == "dummy":
            wrapper_cls = DummyFeatureWrapper
        elif feature == "polynomial":
            wrapper_cls = PolynomialFeatureWrapper
        elif feature == "random":
            wrapper_cls = partial(RandomFeatureWrapper, rand_feat_dim=feature_dim)
        elif feature == "fourier":
            wrapper_cls = partial(FourierFeatureWrapper, rand_feat_dim=feature_dim)
        else:
            raise NotImplementedError("Unsupported feature type")
        # Wrap environment in feature wrapper
        env = wrapper_cls(env)
        # Compute features for dataset
        dataset = FeatureDataset(dataset, env)

    return env, dataset
