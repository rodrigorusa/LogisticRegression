# LogisticRegression
Logistic Regression Python implementation

## Use

Assure if your PC had all of these libs:

pandas / matplotlib / numpy / scipy / scikit-learn / python-tk / python 3.3 or superior

### To compile:

On LogisticRegression folder:

python -m py_compile methods/*.py metric/*.py main.py

### To execute:

python main.py

**usage:** main.py [-h] [-training TRAINING_PATH]
               [-test TEST_PATH] [-plot-confusion-matrix [PLOT_CONFUSION_MATRIX]]
               [-plot-error [PLOT_ERROR]]

Logistic Regression.

  **-h, --help**            show this help message and exit

  **-training (TRAINING_PATH)** - the path of training set CSV file

  **-test (TEST_PATH)** - the path of test set CSV file

  **-plot-confusion-matrix [True|False]** - plot confusion matrix of Logistic Regressors

  **-plot-error [True|False]** - plot training error on Gradient Descent implementations

