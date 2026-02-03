# ETA-DyNN

This is repository contains the code for the paper "**ETA-DyNN: An Energy-Efficient Task Allocation Scheme for Varroa Detection on Edge AI Devices via Dynamic Neural Networks**".

You can find four folders:
- `activity_emulator`: an interactive interface to test the ETA-DyNN approach;
- `environment_aware_task_allocation`: the proposed ETA-DyNN decision-making module;
- `ee_cnn`: the Early-Exit NN;
- `key_frame_extraction`: the algorithm that extracts key frames from the input video.

In order to run any of the code, you first need to
1. install the packages locally through:
```pip install -e .```
2. install the required python dependencies:
```pip install -r requirements.txt```

Then you can move to the folder `activity_emulator` and run the experiment interface with `python main.py`.
