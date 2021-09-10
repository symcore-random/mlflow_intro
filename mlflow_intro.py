# %% initialization
from sklearn.metrics import mean_squared_error
from datetime import datetime
import numpy as np
import mlflow


###########################################
### defining the mlfow logging function ###
###########################################

# This example function mimics a model of e.g. deep learning,
# or catboost. The parameters are like the "learning_rate"
# (for deep learning or catboost) or "max_depth" (for catboost).
# As input for the model, we have an array 'x'.
def model(x, parameters):
    a = parameters["a"]
    b = parameters["b"]
    result = a * x + b
    
    return result


# we define this function for the mlflow logging, given the input data,
# the true output data, and the model parameters. In the end, we log the
# model parameters and the resulting metrics.
def mlflow_parameters_logging(
    input_data, true_output_data, parameters: dict = {"a": 2, "b": 1}
):
    # model prediction, given input data and the model parameters
    prediction = model(input_data, parameters)
    # calculate the mean squared error between the prediction and
    # the true output data
    mse = mean_squared_error(true_output_data, prediction)

    # We define the run name as the date and hour of the run
    # This is important for later identification of best runs
    name = f"{datetime.now().strftime('%d/%b/%Y, %Hh%Mmin%Ss')}"
    # parameter and metric logging
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(parameters)
        mlflow.log_metrics({"mean squared error": mse})


# %%
#################################
### mlflow parameters logging ###
#################################

# parameters trial
parameters = {"a": 2.0, "b": 0.8}
# input data
X = np.array([1, 2, 3, 4, 5, 6])
# true output data
y = np.array([3.1, 4.7, 7.2, 9.02, 11.1, 13.8])
# logging in practice
mlflow_parameters_logging(input_data=X, true_output_data=y, parameters=parameters)

# In order to observe the results, run the following
# command at your terminal (while being just outside
# the mlruns/ directory):
# mlflow ui

######################################################
### I believe there is much more to mlflow than    ###
### what we have explored in this simple exercise, ###
### however this is sufficient for starters.       ###
######################################################