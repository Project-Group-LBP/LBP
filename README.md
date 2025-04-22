# LBP

<!-- Structure of rl_algos/

```
├── ddqn.py
├── ddqn_modified.py
├── deep_sarsa.py
├── dqn.py 
├── ma-sarsa.py
├── maddpg.py 
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
``` -->

## Usage Instructions

- Clone the repository.
- **Create a virtual environment and activate it.**
- Install requirements using `pip install -r requirements.txt`.

### Training

To train the MADDPG model:

```bash
cd multi_uav_coverage_maddpg

python train.py --num_episodes=1000

python train.py --resume="saved_models/maddpg_episode_100" --num_episodes=1000
```

### Testing

To test a trained model:

```bash
cd multi_uav_coverage_maddpg

python test.py --model_path="saved_models/maddpg_episode_final" --num_episodes=50

```

---


<!-- Testing Results :

Best performing model (run for 50 episodes) : 
```
Average values over all episodes:
reward: -11771.382959
coverage: 0.330369
fairness: 0.331693
energy_efficiency: 0.135578
time: 117.566891
penalty: 1413.680000
``` -->