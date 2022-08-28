# :computer: Models Code

This repository contains the code base to deal with the data set that was collected
from Kinderlabor exercise (types). It can be used to analyse the data set as well as to train (DL) models on it.

# :open_file_folder: Code structure

So far, the code has these main parts:

- data_augmentation.py: The Data Augmentation utils allowing to run different Data Augmentation strategies (all of them
  building on a 32x32 grayscale image)
- data_loading.py: The Data Loader class, which reads the CSV file, splits it into
  the three sets (training, validation, test sets) and then moves the images to the respective folder
- grayscale_model.py: A neural network that operates on a grayscale image of size 32x32, inspired by the VGG model
- run_xx: Different run scripts
- training.py: The trainer class, which can train a model based on given data loaders and then predict on the test set
- visualizing.py: The visualization class to visualize some samples, the training progress and confusion matrix

# :floppy_disk: Setup

- Install Python 3.8 or higher (check https://www.python.org/)
- Create a virtual environment, e. g. with venv and install all the requirements stated in requirements.txt file (
  see https://docs.python.org/3/library/venv.html)
- Install PyTorch 1.11 or higher as stated in their docs: https://pytorch.org/get-started/locally/
- You are ready to go :sunglasses:

# :rocket: Herby Integration

- Script model (see trainer class) with torch jit
- Load model using DJL (https://djl.ai/) with the according synset (class names) from disk and predict on image
- About scripting transforms as well: see https://github.com/deepjavalibrary/djl/issues/1556, works by adding
  environment variable PYTORCH_EXTRA_LIBRARY_PATH=python3.9/site-packages/torchvision/_C.so (.pyd on Windows)

# :ledger: TODOs

- Update experiments that compare different data augmentations from following options. Limit combinations from those:
    - scaling (std vs only min-max in range [[0, 1]])
    - inverting
    - contrast
    - rotate / shift / scale
    - center crop for pre-processing comparisons?
    - Sheer?
- Implement further data splits with semantic meaning (keeping out students, respectively classes in test set)
  once more data is labelled, maybe include some real samples in training and see if it is better
- Open Set Experiments
    - track softmax probabilities and plot them
    - Add background class possibility
- Distinguish basic and extended command exercises and see whether model is better when having limited class space
- Minor code adaptions
    - Fix herby not correcting known rotation
    - Actually visualization of dataset could be stored per split and not per run (class distribution and some samples)
    - Drop task (12e-f), where version problem was present and evtl 16c where bounding box was badly, alternatively make
      label bounding box error (shortcut b) in Herby