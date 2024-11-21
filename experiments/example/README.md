# Example Project: Speaker Identification Model on LibriSpeech

This repository contains an example experiment for training a speaker identification model on the LibriSpeech dataset. 
The project is structured to be extensible and leverages Hydra for configuration management, support for distributed training with PyTorch, and modularized components for data preparation, training, and evaluation.

## Project Structure
```
example/
├── configs/             # Configuration files for the experiment.
│   ├── augment/         # Augmentation-related settings.
│   ├── data/            # Data-related settings.
│   ├── hydra/           # Hydra-specific configurations.
│   ├── job/             # Job-related settings (e.g., device setup).
│   ├── loss/            # Loss function configurations.
│   ├── model/           # Model-related settings.
│   ├── optimizer/       # Optimizer configurations.
│   ├── scheduler/       # Learning rate scheduler settings.
│   ├── training/        # Training hyperparameters.
│   ├── wandb/           # Weights & Biases integration settings.
│   └── config.yaml      # Main configuration entry point.
├── __init__.py          # Package initialization.
├── augmentation.py      # Data augmentation utilities.
├── data_loading.py      # Data loader setup.
├── data_preparation.py  # Dataset preparation script.
├── evaluator.py         # Evaluation logic and metrics.
├── model.py             # Model architecture definition.
├── run.py               # Main experiment script.
├── trainer.py           # Training logic and utilities.
└── README.md            # Project documentation.
```

## How to Run the Experiment
### 1. Setup
Clone the repository and install the required dependencies:
```bash
git clone "https://github.com/HolgerBovbjerg/aisnet.git"
cd aisnet
```

It is recommended to create a virtual environment before installing packages.  
For example:
```bash
python3 -m pip install --upgrade pip
python3 -m venv venv
source venv/bin/activate
```
The dependencies can then be installed via:
```
pip install -r requirements.txt
```

### 2. Add project root to ```PYTHONPATH```

This ensures that the project root is part of ```PYTHONPATH```.
```bash
export PYTHONPATH=".":$PYTHONPATH 
```

## Run experiment

### Training Script Overview

The experiment logic is defined in ```run.py```.

Inside run.py a function:
```
def run(config, distributed: bool = False):
    some_experiment_logic...
```

All experiments should have a ```run.py```file with a ```run``` function defined inside.
This is necessary as this is what train.py will look for when running the experiment. 

The ```run.py``` script can be structured in any way as long as it accepts a hydra config, ```config```, and a boolean flag, ```distrbuted```, as input.
The ```distributed``` flag is derived from the configuration and simply tells whether the experiment will use distributed training.

For this example project the experiment is divided into three stages inside ```run.py```.
- Stage 0: Data Preparation 
- Stage 1: Model Training 
- Stage 2: Evaluation

In stage 0, data will be downloaded and relevant files will be prepared for the experiment.

In stage 1, the model will train for a number of epochs.

In stage 2, the model will be evaluated on a test set.

### Configuration Management
Hydra is used for managing configurations.

The main configuration is found in ```configs/config.yaml```.
This file defines the default configuration of the experiment.
In theory the configuration file could be any .yaml file.
In our case, we are using a Trainer module based on source.trainer.BaseTrainer,
which expects certain entries in the configuration file.

Although we could have one big configuration files, the configuration in this example is modular such that each configuration area has its own module.
This makes it easier to organize configurations for large experiments, with many settings.
For instance, ```configs/job/default.yaml``` contains settings related to the job such as number of workers, device (cpu/gpu), experiment name etc.

Besides the configuration modules ```config.yaml``` has the following settings:
* experiment_type: "example"
* exp_dir: <path_to_where_experiment_outputs_are_saved>
* exp_name: name_of_experiment
* experiment_type: "doa"
* start_stage: 0 
* stop_stage: 100 
* data_path: </path/to/data/folder>

The ```experiment_type``` setting defines the type of experiment you are running.
This changes where ```train.py``` looks for a runscript ```run.py```.
For example, when specifying ```example``` train.py will try to import ```experiments.example.run```.

The entry ```exp_dir``` is used to determine where the outputs from the experiment, 
such as checkpoints and logs, are saved.

The ```exp_name``` is the experiment identifier, and will also be used to identify the experiment on wandb if used.

The ´´´start_stage´´´and ´´´stop_stage´´´ options are used in run.py to define which stages to run.
Notice that the script only has two stages while the default stop stage is 100.
In this case the run script will simply finish when no more stages are defined.

The parameter ```data_path``` is used to define where the data used in the experiment will be stored and loaded from.

#### Stage 0: Data preparation
In the case of this experiment, the data can be automatically downloaded. 
However, the script will check if the data already exists in the specifed data_path and in this case skip downloading the data.
If it does not already exist, the script will check ```data/database.yaml``` to see if the data set is defined here. 
If the data set is not defined in ```<aisnet_root>/data/database.yaml``` an error will be thrown.
Therefore, it can be necessary to update ```data/database.yaml``` if you wish to use a new dataset.

The file ```<aisnet_root>data/database.yaml``` stores paths to data sets used in experiments.
For each data set name, one can either specify ```download``` or set a path ```/path/to/data_set/root/```.
* If download is specified, the data set will be downloaded to ```data_path``` defined in the configuration.
* If a path is specified, the script will copy the files to ```data_path```. 
* If the specified path is a subfolder to ```data_path``` the script will do nothing.
* If the data set the data set is not in the database or is specified without a path, you will be prompted to put either ```download``` or a path to the data set root.

#### Stage 1: Model training
In this stage the model will be built using the settings specified in the configuration.
Additionally, data loaders will be setup, and the environment for distributed training will also be configured if specified.
These will be passed to a ```trainer``` module defined in ```trainer.py``` and the model will be trained.

#### Stage 2: Model evaluation
In the evaluation stage, the trained model will be loaded from a checkpoint and evaluated using the ```evaluator``` defined in ```evaluator.py```.

#### Customizing experiment configuration
**Commandline**: 
The default configuration can be overridden via the command line or by editing files in the configs/ directory.

For example, to override the default config from the commandline and only run data preparation we can run:
```bash
python cli/train.py start_stage=0 stop_stage=0
```

We can also change the training set from 100h Librispeech training set to the full 960h Librispeech training data using the following command:
```bash
python cli/train.py data/train=librispeech_train_960h model.num_classes=2338
```

Here we override the default train data set config ```librispeech_train_100h``` with ```librispeech_train_960h```.
Additionally, because we now have more speakers, we change the number of output classes from ```num_classes=251``` to ```num_classes=2338``` to accomodate the additional number of speakers.

**Custom config files**
Instead of using commandline overrides, we could also create a new config file, e.g., ```my_custom_config.yaml```, with a custom configuration, and override the default config name via commandline:
```bash
python cli/train.py --config_name my_custom_config
```
The custom config should be placed inside the configs folder at the same level as ```config.yaml```.
The default ```config_name``` is ```config```.  

Alternatively, you can create a new folder for custom configs, e.g., ```my_configs```, and point the script to look in this folder via the ```--config_path``` argument.
```bash
python cli/train.py --config_path </path/to/custom_configs/> --config_name my_custom_config.yaml
```

The argument given to ```--config_path``` should be an absolute path or a path relative to cli/train.py. 
The default value is ```../experiments/example/configs```.

### Running other experiments
Let us say we have another experiment ```example2``` with a configuration file using the default config file name ```config.yaml``` inside the folder ```../experiments/example2/configs```.
We could run this experiment by overriding the default config path as follows:

```bash
python cli/train.py --config_path ../experiments/example2/configs
```


