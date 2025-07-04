# LBP

## Multi-Agent Deep Reinforcement Learning for Coverage Maximization

Structure of rl_algos/

```
├── ddqn.py
├── ddqn_modified.py
├── deep_sarsa.py
├── dqn.py 
├── ddpg.py
├── ma-sarsa.py
├── maddpg.py 
├── ppo.py 
├── q_learning.py
└── sarsa.py 
```

Structure of multi_uav_coverage_maddpg/

```
├── env.py
├── test.py
├── train.py
├── utils/
│   ├── data_points.py
│   ├── decoded_points.py
│   ├── input.py
│   ├── logger.py
│   └── plot_logs.py
└── maddpg/
    ├── agents.py
    ├── buffer.py
    ├── cnn.py
    └── maddpg_uav.py
```

## Usage Instructions

- Clone the repository : `git@github.com:Project-Group-LBP/LBP.git` .
- **Create a virtual environment and activate it.**
- Install requirements using `pip install -r requirements.txt`.

### Training

To train the MADDPG model:

```bash
cd multi_uav_coverage_maddpg

# Basic training with default settings (500 episodes)
python train.py

# Train with custom number of episodes
python train.py --num_episodes=1000

# Train using image initialization
python train.py --use_img --img_path="path/to/image.png"

# Resume training from saved model
python train.py --resume="saved_models/maddpg_episode_100" # can input pending no of episodes
```

### Testing

To test a trained model:

```bash
cd multi_uav_coverage_maddpg

# Basic testing with default settings (50 episodes)
python test.py --model_path="saved_models/maddpg_episode_final"

# Test with custom number of episodes
python test.py --model_path="saved_models/maddpg_episode_final" --num_episodes=25

# Test with image initialization
python test.py --model_path="saved_models/maddpg_episode_final" --use_img --img_path="path/to/image.png"
```

---
