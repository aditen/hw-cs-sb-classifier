# :computer: Models Code

This repository contains the code base to deal with the data set that was collected
from Kinderlabor exercise (types). It can be used to analyse the data set as well as to train (DL) models on it.

# :open_file_folder: Code structure
So far, the code has three main parts:
- data_loading.py: The Data Loader class, which reads the CSV file, splits it into 
the three sets (training, validation, test sets) and then moves the images to the respective folder
- grayscale_model.py: A neural network that operates on a grayscale image of size 32x32, inspired by the VGG model
- run_xx: Different run scripts
- training.py: The trainer class, which can train a model based on given data loaders and then predict on the test set
- visualizing.py: The visualization class to visualize some samples, the training progress and confusion matrix

# :floppy_disk: Setup
- Install Python 3.8 or higher (check https://www.python.org/)
- Create a virtual environment, e. g. with venv and install all the requirements stated in requirements.txt file (see https://docs.python.org/3/library/venv.html)
- Install PyTorch 1.11 or higher as stated in their docs: https://pytorch.org/get-started/locally/
- You are ready to go :sunglasses:

# :ledger: TODOs
- Implement further data splits with semantic meaning (keeping out students, respectively classes in test set)
- Further visualizations: allow to grasp through wrong classifications
- Minor code adaptions 
  - Data Augmentation: Invert grayscale (such that written signs have actually high values and not background) 
  - Better title for sample displaying
  - Allow *rotation* data augmentation for commands and crosses (but not orientation!)
  - Allow reusage of image folders if present and make force_recreation parameter in constructor to force dataset recreation
- Maybe experiment with cropping of images, if cropping over the border helps (questionable)
- Start experimenting with known/unknown unknowns
- Start evaluating identical base models for different tasks (probs needs large code adaptations!)