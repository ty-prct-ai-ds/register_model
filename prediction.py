import mlflow.pyfunc
import pandas as pd
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

data = pd.DataFrame({
    'ph': 3.71608,
    'Hardness': 204.89045,
    'Solids': 20791.31898,
    'Chloramines': 8.0,
    'Sulfate': 368.516,
    'Conductivity': 564.308,
    'Organic_carbon': 10.3798,
    'Trihalomethanes': 86.99,
    'Turbidity': 2.96,
},index=[0])

logged_model = 'runs:/4b8e8dcbcb7241fd8b2f7cd502457b69/Best Model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

print(loaded_model.predict(pd.DataFrame(data)))
