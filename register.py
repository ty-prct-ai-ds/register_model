from mlflow.tracking import MlflowClient
import mlflow

client = MlflowClient()

run_id = "4b8e8dcbcb7241fd8b2f7cd502457b69"

model_path = "mlflow-artifacts:/863203015088888540/4b8e8dcbcb7241fd8b2f7cd502457b69/artifacts/Best Model"

model_name = "water_potability_rf"



model_uri = f"runs:/{run_id}/{model_path}"

mlflow.register_model(model_uri,model_name)