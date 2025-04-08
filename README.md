# LBP

```
├── env.py
├── test.py
├── train.py
├── logs_and_inputs/
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

```bash
python multi_uav_coverage/train.py # run using coordinates in data_points.py (no image input)
# Or use
python multi_uav_coverage/train.py --use_img --img_path="path_of_img_relative_to_current_dir" # (for image input)
```

Similar syntax for `test/py`