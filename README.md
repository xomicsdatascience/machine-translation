# machine-translation
Replication of the "[Attention Is All you Need](https://arxiv.org/abs/1706.03762)" machine translation model using [AttentionSmithy](https://github.com/xomicsdatascience/AttentionSmithy), a package for creating transformer models.

# Main Files
## scripts/0_data_prep.py
This file downloads the WMT-14 German-English dataset and processes it for loading into the model. This is also where the train/val/test split occurs. 

Each dataset (train/val/test) consists of two files, one for English (en) and one for German (de), matched by line index. For example, line 5 of `train_en.txt` is the English translation of line 5 of `train_de.txt`, which consists of German text.

The loaded dataset consists of sentences. This script converts those sentences into tokens, then adds them as a comma-delimited line to the relevant file.

## scripts/1_train_model.py
Has much in common with the file `scripts/model_script_for_nas.py`, which was specific to use with a neural architecture search (NAS). This script assembles and trains the machine translation model. There are several arguments to be used with the script - below is an example usage.

`python 1_train_model.py --loss_type custom --label_smoothing 0.9 --embed_dim 512 --dim_feedforward 2048 --number_of_layers=6`

## src/machine_translation/MachineTranslationModel.py
The code for the model used in machine translation. It was written using pytorch lightning for readability, and thus outlines the construction of the model, the forward pass process, and how that looks for training and validation steps.

## src/machine_translation/data/MachineTranslationDataModule.py
The code for preparing the data module used in training and validating the machine translation model. It is made to be used with the pytorch lightning Trainer class, as called in model training scripts.

# Additional Files for interested readers
## scripts/run_nas.py
This code runs a neural architecture search (NAS). The code is based on the [Multi-Objective NAS with Ax](https://pytorch.org/tutorials/intermediate/ax_multiobjective_nas_tutorial.html) tutorial, and calls the `scripts/model_script_for_nas.py` in each pass with new parameters selected during the search.

## src/machine_translation/data/LineIndexDataset.py
This code is used to extract specific lines from train, val or test datasets when forming a batch. Using this class allows the user to reference data efficiently without holding the entire dataset in memory.

## src/machine_translation/data/LengthBatchSampler.py
This code groups samples together by context window length for efficient training. A similar strategy was employed in the original Attention Is All You Need paper.
