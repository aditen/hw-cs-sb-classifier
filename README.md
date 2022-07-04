# Models Code :telescope:

This repository contains the code base to deal with the data set that was collected
from Kinderlabor exercise (types). It can be used to analyse the data set as well as to train (DL) models on it.

# Code structure :cd:
So far, the code has three main parts:
- data_loading.py: The Data Loader class, which reads the CSV file, splits it into 
the three sets (training, validation, test sets) and then moves the images to the respective folder
- grayscale_model.py: A neural network that operates on a grayscale image of size 32x32, inspired by the VGG model
- run_xx: Different run scripts
- training.py: The trainer class, which can train a model based on given data loaders and then predict on the test set
- visualizing.py: The visualization class to visualize some samples, the training progress and confusion matrix

# Setup :floppy_disk:
- Install Python 3.8 or higher (check https://www.python.org/)
- Create a virtual environment, e. g. with venv and install all the requirements stated in requirements.txt file (see https://docs.python.org/3/library/venv.html)
- Install PyTorch 1.11 or higher as stated in their docs: https://pytorch.org/get-started/locally/
- You are ready to go :sunglasses:

# TODOs :page_with_curl:
- Implement data splits with semantic meaning (keeping out students, classes in test set)
- Clean up code, update Readme to explain implemented object model
