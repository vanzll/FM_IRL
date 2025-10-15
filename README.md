# FM-IRL: Flow-Matching for Reward Modeling and Policy Regularization in Reinforcement Learning

<h1 align="center"> 
    <img src="./framework_detailed.png" width="1000"><img src="./framework_overview.png" width="1000">
</h1>

The Official implementation of [**FM-IRL: Flow-Matching for Reward Modeling and Policy Regularization in Reinforcement Learning**](https://arxiv.org/abs/2510.09222).

Zhenglin Wan\*<sup>1</sup>,
Jingxuan Wu\*<sup>2</sup>,
Xingrui Yu<sup>3</sup>,
Chubin Zhang<sup>4</sup>,
Mingcong Lei<sup>5</sup>,
Bo An<sup>1</sup><br>, Ivor Tsang<sup>3</sup><br>
<sup>1</sup>Agent Mediated Lab, NTU, Singapore
<sup>2</sup>Department of Statistics and Operations Research, UNC-Chapel Hill, America <sup>3</sup>CFAR, A*STAR, Singapore<br> <sup>4</sup>School of Computer Science, BUPT, China <br> <sup>5</sup>School of Data Science, CUHK(SZ), China

(\*Equal contribution)


This work proposes a novel framework that addresses the limitations of Flow Matching (FM) in online settings by introducing a student-teacher architecture. While FM excels at offline behavioral cloning, it struggles with exploration and online optimization due to gradient instability and high inference costs. Our approach employs a simple MLP-based student policy for efficient environment interaction and online RL updates, guided by a reward model derived from a teacher FM policy that encapsulates expert distribution knowledge. The teacher FM simultaneously regularizes the student's behavior to stabilize learning. This combination maintains the expressiveness of FM while enabling stable gradient computation and efficient exploration, significantly improving generalization and robustness, particularly with suboptimal expert data.

## Environment Setup and Installation Guide

### Software Environment Configuration

This codebase requires `Python 3.8` or higher. All required packages are listed in the `requirements.txt` file. To set up the environment from scratch using Anaconda, execute the following commands:
   ```   
   conda create -n [your_env_name] python=3.8
   conda activate [your_env_name]
   ./utils/setup.sh
   ```

### Weights & Biases Setup

Configure [Weights and Biases](https://wandb.ai/site) by first logging in with `wandb login <YOUR_API_KEY>` and then editing `config.yaml` with your W&B username and project name.

## Expert Demonstration Setup

Expert demonstration data is stored in the *expert_datasets* folder. Use Git LFS to access the expert demonstration files.

## Experiment Reproduction Guide

To replicate the experiments conducted in our paper, follow these steps:

### 1. Select Configuration Files
The wandb sweep configuration files for all tasks can be found in the `configs` directory. Each subdirectory (e.g., `./configs/ant`) contains at least seven common files:
- `airl.yaml`
- `diffusion_policy.yaml`
- `drail.yaml`
- `fm_policy.yaml`
- `fmirl.yaml`
- `gail.yaml`
- `wail.yaml`

### 2. Run Experiments
After selecting the desired configuration file, execute the following command:
   ```
   ./utils/wandb.sh <Configuration_file_path.yaml>
   ```

The results will be stored in ./data/log




## Code Structure Overview

### Core Components
- `fmirl`: Implementation of our main method
- `utils`: Utility scripts
  - `utils/wandb.sh`: Script to automatically create and execute wandb commands from configuration files
  - `utils/setup.sh`: Script to install and set up the conda environment
- `shape_env`: Custom environment code for reference
- `goal_prox`: Customized environment code from [goal_prox_il](https://github.com/clvrai/goal_prox_il)
  - `goal_prox/envs/ant.py`: AntGoal locomotion task
  - `goal_prox/envs/fetch/custom_fetch.py`: FetchPick task
  - `goal_prox/envs/hand/manipulate.py`: HandRotate task
- `rl-toolkit`: Base RL code and imitation learning baselines from [rl-toolkit](https://github.com/ASzot/rl-toolkit)
  - `rl-toolkit/rlf/algos/on_policy/ppo.py`: PPO policy updater code for RL
  - `rl-toolkit/rlf/algos/il/gail.py`: Baseline Generative Adversarial Imitation Learning (GAIL) code
  - `rl-toolkit/rlf/algos/il/wail.py`: Baseline Wasserstein Adversarial Imitation Learning (WAIL) code
  - `rl-toolkit/rlf/algos/il/dp.py`: Baseline Diffusion Policy code
- `d4rl`: Codebase from [D4RL: Datasets for Deep Data-Driven Reinforcement Learning](https://github.com/rail-berkeley/d4rl) for Maze2D

## Acknowledgements

### Code Sources
- Base code adapted from [goal_prox_il](https://github.com/clvrai/goal_prox_il) and [DRAIL](https://www.bing.com/search?q=DRAIL&qs=n&form=QBRE&sp=-1&ghc=1&lq=0&pq=dra&sc=12-3&sk=&cvid=5963FB49AC6B4B4695432D9D97013E75)
- Fetch-pick and Hand-rotate environments customized based on [OpenAI](https://github.com/openai/robogym) implementations
- Ant environment customized by [goal_prox_il](https://github.com/clvrai/goal_prox_il) and originated from [Farama-Foundation](https://github.com/Farama-Foundation/Gymnasium)
- Maze2D environment based on [D4RL: Datasets for Deep Data-Driven Reinforcement Learning](https://github.com/rail-berkeley/d4rl)

## Citation

If you find this work useful, please consider to give a star and cite our paper:

```bibtex
@misc{wan2025fmirlflowmatchingrewardmodeling,
      title={FM-IRL: Flow-Matching for Reward Modeling and Policy Regularization in Reinforcement Learning}, 
      author={Zhenglin Wan and Jingxuan Wu and Xingrui Yu and Chubin Zhang and Mingcong Lei and Bo An and Ivor Tsang},
      year={2025},
      eprint={2510.09222},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.09222}, 
}
```
