# LBP

Structure of rl_algos/

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
```

---

for `train.py` :

```bash
cd multi_uav_coverage_maddpg
python train.py # run using coordinates in data_points.py (no image input)
# Or use
python train.py --use_img --img_path="path of img relative to current_dir" # (for image input)
```

for `test.py` : 

```bash
cd multi_uav_coverage_maddpg
python test.py --model_path="directory where models to be loaded are stored relative to current dir"
```

---
