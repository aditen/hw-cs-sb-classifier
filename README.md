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
- utils.py: Project-specific Utilities

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

# :mag: Reproducibility
In order to reproduce the experiments, seeds as well as deterministic convolutions were used on an Nvidia 1050 TI
Set CUBLAS_WORKSPACE_CONFIG=:4096:8 in your environment - otherwise the experiments throw an error or may be different

# :ledger: TODOs

- Minor code adaptions
    - In data collection methodology comparison:
        - Add table with classes and number of samples
        - Add second table with splits and number of samples for appendix
    - Larger font in plots