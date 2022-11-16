# Implicit Neural Representations
This project was created for the Master Thesis at Warsaw University of Technology with topic "Analysis of the use of neural networks as an implicit representation of sound".

### Data
In this project I have analyzed three different types of sound data:
+ Voice messages (VCTK)
+ Music (GTZAN)
+ Environmental sounds (ESC-50)

### Run
In order to run set up the config file based on the ones that are available in `cfg/` directory and run `python main.py --config-file <PATH_AND_NAME_OF_CONFIG_FILE.yaml`.

All possible attributes for the project are listed in `cfg/__init__.py` file in `MainConfig` dataclass.

### Install
You can install all needed packages with command `pip3 install -e <PATH_TO_PROJECT>`.
If you are running command from inside the sound_representation repository you can just run `pip3 install -e .`.
