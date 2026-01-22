# ETA-DyNN

This is repository contains the code for the paper "**ETA-DyNN: An Energy-Efficient Task Allocation Scheme for Varroa Detection on Edge AI Devices via Dynamic Neural Networks**".

You can find two folders:
- `activity_emulator`: contains the code to run an interactive interface to test the ETA-DyNN approach;
- `environment_aware_task_allocation`: contains the code of the proposed ETA-DyNN decision-making module.

In order to run any of the code, you first need to
1. install the packages locally through:
```pip install -e .```
2. install the required python dependencies:
```pip install -r activity_emulator/requirements.txt```

Then you can move to the folder `activity_emulator` and run the experiment interface with `python main.py`.
