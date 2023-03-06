# SINDy: Sparse Identification of Non-linear Dynamical Systems

SINDy-Model-Prediction

#Overview
This repository contains Python code for training a SINDy model for different dynamical systems, along with a set of example models that can be used to demonstrate the capabilities of the code.

Model: The dynamical system model is defined in the Model section of the script. The current version includes four example models: cubic_2D, lorenz, quadratic, and linear_2D.

#The code requires the following Python libraries:
numpy
torch
matplotlib
pandas
pysindy
scipy



Data: The script generates data by solving the differential equations of the dynamical system model using the solve_ivp function from scipy.integrate. The generated data is then preprocessed, split into training and testing datasets, and used to train the SINDy model.
Preprocess: The data is preprocessed using MinMaxScaler to scale the time values to the range [-1,1].
Split: Train-Test: The preprocessed data is then split into a training dataset and a testing dataset. The first 600 time steps are used for training, and the rest are used for testing.
Model: A SINDy model is created using the ps.SINDy function from the pysindy library. The model is fitted to the training data using a polynomial feature library and the STLSQ optimizer.
Data-derivatives: The SINDy model is then used to predict the time derivatives of the data, both for the training and testing datasets.
Data from model-output: The SINDy model is then used to predict the future time steps for the testing data. The predicted data is compared to the actual data to evaluate the prediction score.
Requirements
Python 3.x
NumPy
PySINDy
PyTorch
SciPy
Scikit-learn
Matplotlib
Usage
Clone the repository to your local machine.
Open the script SINDy_Model_Prediction.py using a Python editor or IDE.
Choose the dynamical system model you want to use by uncommenting it in the Model section of the script. By default, linear_2D is selected.
Run the script. The results will be displayed in two plots: one for the dataset and its predicted values, and another for the time derivatives and their predicted values.
Credits
The SINDy model is implemented using the pysindy library. The script is adapted from the tutorial on SINDy by the authors of the library.
