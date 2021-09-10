# %% initialization
from sklearn.metrics import mean_squared_error
from datetime import datetime
import numpy as np
import mlflow


###########################################
### defining the mlfow logging function ###
###########################################


def model(x, parameters):
    a = parameters["a"]
    b = parameters["b"]
    result = a * x + b

    return result


def mlflow_parameters_logging(
    input_data, true_output_data, parameters: dict = {"a": 2, "b": 1}
):

    prediction = model(input_data, parameters)
    mse = mean_squared_error(true_output_data, prediction)

    # Define the run name
    name = f"{datetime.now().strftime('%d/%b/%Y, %Hh%Mmin%Ss')}"
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(parameters)
        mlflow.log_metrics({"mean squared error": mse})


# %%
#################################
### mlflow parameters logging ###
#################################

parameters = {"a": 2.0, "b": 0.8}
X = np.array([1, 2, 3, 4, 5, 6])
y = np.array([3.1, 4.7, 7.2, 9.02, 11.1, 13.8])

mlflow_parameters_logging(input_data=X, true_output_data=y, parameters=parameters)

#%%
