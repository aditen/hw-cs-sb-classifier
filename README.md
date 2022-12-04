# :computer: Models Code

This repository contains the code base to deal with the data set that was collected
from Kinderlabor exercise (types). It can be used to analyse the data set as well as to train (DL) models on it.

# :open_file_folder: Code structure

So far, the code has these main parts:

- data_augmentation.py: The Data Augmentation utils allowing to run different Data Augmentation strategies (all of them
  building on a 32x32 grayscale image)
- data_loading.py: The Data Loader class, which reads the CSV file, splits it into
  the three sets (training, validation, test sets) and then moves the images to the respective folder
- grayscale_model.py: Implementation of adapted LeNet-5 and SimpleNet models to operate on a single channel (grayscale)
  image of size 32x32
- open_set_loss.py: The implementation of the open set loss functions (Entropic Open Set, Binary EOS)
- running.py: The implementation of the experiments, each as a separate function
- run_script.py: Run Script providing different flags to reproduce experiments
- training.py: The trainer class, which can train a model based on given data loaders and can be used to predict on the
  test set
- utils.py: Project-specific Utilities and types (mainly enumerations)
- visualizing.py: The visualization class to visualize some samples, the training progress, confusion matrix and more

# :floppy_disk: Setup and Use

- Install Python 3.8 or higher (check https://www.python.org/)
- Create a virtual environment, e. g. with venv and install all the requirements stated in requirements.txt file (
  venv is documented here: https://docs.python.org/3/library/venv.html). The command to install the requirements is *pip install -r requirements.txt*
  If there are problems with the installation of torch, follow the instructions from the webpage: https://pytorch.org/get-started/locally/
- Activate the virtual environment and you are ready to go :sunglasses:

Run ``python ./run_script.py --help`` from the directory of the code to see all commands the run script supports.

*General Hint: The author used PyCharm Professional and the scientific view of Matplotlib. The layout of charts can differ when using a different setup or not work entirely*

*Hint 1: The usage of the --no-plot-windows flag is highly recommended when training the baseline experiment!*
Otherwise, many windows are opened. Also, if you do not use this flag then the visualizations on disk may be stored with the odd window tight layout

*Hint 2: If you get an error message regarding determinism, check out the Section 'Reproducibility' of this README file*

# :rocket: Herby Integration

- Script model (see trainer class) with torch jit
- Load model using DJL (https://djl.ai/) with the according synset (class names) from disk and predict on image
- About scripting transforms as well: see https://github.com/deepjavalibrary/djl/issues/1556, works by adding
  environment variable PYTORCH_EXTRA_LIBRARY_PATH=python3.9/site-packages/torchvision/_C.so (.pyd on Windows)

# :abc: Naming

Generally, the code tries to use the same naming as the thesis. Sometimes, there are slight deviations. For example the following ones:

- OSCR = OSRC (Open Set Classification Rate Curve)
- Instruction vs Command for the Task Type and Cross vs Checkbox for the Task Type

# :mag: Reproducibility

In order to enable reproducibility of the experiments, seeds as well as deterministic convolutions were used on an
Nvidia 1050 TI

Set the variable CUBLAS_WORKSPACE_CONFIG=:4096:8 in your environment if you use CUDA. 
Otherwise, the experiments throw a Cublas Error.

If reproducibility is not that important, you can go to the utils.py file and comment out the following line:
torch.use_deterministic_algorithms(True)

# :ledger: TODOs
* May originate from the future work of the thesis (error analysis on student level etc.)
