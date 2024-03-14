## Transferable Reinforcement Learning via Generalized Occupancy Models

####  [[Website]](https://weirdlabuw.github.io/gom/) [[Paper]](https://arxiv.org/abs/2403.06328) 

[Chuning Zhu<sup>1</sup>](https://homes.cs.washington.edu/~zchuning/), [Xinqi Wang<sup>1</sup>](https://elliotxinqiwang.github.io/), [Tyler Han<sup>1</sup>](https://thanandnow.github.io/), [Simon Shaolei Du<sup>1</sup>](https://simonshaoleidu.com/), [Abhishek Gupta<sup>1</sup>](https://homes.cs.washington.edu/~abhgupta/)<br/>

<sup>1</sup>University of Washington

This is a Jax implementation of Generalized Occupancy Models (GOMs). GOM is an unsupervised reinforcement learning method that models the distribution of all possible outcomes represented as discounted sums of state-dependent cumulants. The outcome model is paired with a readout policy that produces an action to realize a particular outcome. Assuming a linear dependence of rewards on cumulants, transferring to downstream tasks reduces to performing linear regression and solving a simple optimization problem for the optimal possible outcome. 

## Instructions

#### Setting up repo
```
git clone https://github.com/WEIRDLabUW/gom
```

#### Install Dependencies
```
pip install -r requirements.txt
```


## D4RL Experiments
To train GOMs on D4RL datasets and adapt to the default tasks, run the following commands
```
# Antmaze
python train.py env_id=antmaze-umaze-v2 exp_id=benchmark seed=0
python train.py env_id=antmaze-umaze-diverse-v2 exp_id=benchmark seed=0
python train.py env_id=antmaze-medium-diverse-v2 exp_id=benchmark seed=0
python train.py env_id=antmaze-medium-play-v2 exp_id=benchmark seed=0
python train.py env_id=antmaze-large-diverse-v2 exp_id=benchmark seed=0
python train.py env_id=antmaze-large-play-v2 exp_id=benchmark seed=0

# Kitchen
python train.py --config-name atrl_kitchen.yaml env_id=kitchen-partial-v0 exp_id=benchmark seed=0
python train.py --config-name atrl_kitchen.yaml env_id=kitchen-mixed-v0 exp_id=benchmark seed=0
```

To adapt a trained GOM to a new downstream reward, relabel the dataset with the new reward function (e.g. by adding a env wrapper and modifying the dataset class) and run the following command (changing `env_id` correspondingly)
```
python train_w.py env_id=antmaze-medium-diverse-v2 exp_id=benchmark seed=0
```
This will load the pretrained outcome model and readout policy and perform linear regression to fit the new rewards.

## Preference antmaze experiments
To run the preference antmaze experiments, install D4RL with the custom antmaze environment from this [repository](https://github.com/zchuning/D4RL). Then, download
the accompanying dataset from this [link](https://drive.google.com/file/d/1msNLNNx35wr8fwPNXev0GnTzqoOGjYBx/view?usp=sharing) and place it in `data/d4rl` under the project root directory. Run the following commands to train on each preference mode. Alternatively, train on only one mode and adapt to the other mode using the adaptation script.

```
# Go Up
python train.py env_id=multimodal-antmaze-0 exp_id=benchmark seed=0 planning.planner=random_shooting
# Go Right
python train.py env_id=multimodal-antmaze-1 exp_id=benchmark seed=0 planning.planner=random_shooting
```


## Roboverse experiments
To run the roboverse experiments, download the roboverse dataset from this [link](https://drive.google.com/drive/folders/1jxBQE1adsFT1sWsfatbhiZG6Zkf3EW0Q) and place the files `data/roboverse` under the project root directory. Use one of the following commands to train a GOM.
```
python train.py --config-name atrl_roboverse.yaml env_id=roboverse-pickplace-v0 exp_id=benchmark seed=0
python train.py --config-name atrl_roboverse.yaml env_id=roboverse-doubledraweropen-v0 exp_id=benchmark seed=0
python train.py --config-name atrl_roboverse.yaml env_id=roboverse-doubledrawercloseopen-v0 exp_id=benchmark seed=0
```

## Bibtex
If you find this code useful, please cite:

```
@article{zhu2024gom,
    author    = {Zhu, Chuning and Wang, Xinqi and Han, Tyler and Du, Simon Shaolei and Gupta, Abhishek},
    title     = {Transferable Reinforcement Learning via Generalized Occupancy Models},
    booktitle = {ArXiv Preprint},
    year      = {2024},
}
```
