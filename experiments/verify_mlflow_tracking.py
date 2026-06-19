from aegis_hgx.utils.config import load_yaml

from pathlib import Path
import mlflow
from typing import Any


CONFIG_PATH = Path("configs/baseline_logistic.yaml") 

def create_experiment_storage(configs:dict[str, Any]):

    experiment_tracking = configs["experiment_tracking"]
    database_path = Path("mlflow/mlflow.db")
    database_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_root = Path(experiment_tracking["artifact_root"]).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    return database_path, artifact_root


def main() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        configs = load_yaml(CONFIG_PATH)

    database_path, artifact_root = create_experiment_storage(configs)

    mlflow.set_tracking_uri(configs["experiment_tracking"]["uri"])

    experiment=mlflow.get_experiment_by_name(configs["experiment_tracking"]["experiment_name"])
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=configs["experiment_tracking"]["experiment_name"], 
            artifact_location=artifact_root.as_uri()
            )
    else:
        experiment_id=experiment.experiment_id

    mlflow.set_experiment(experiment_id=experiment_id)

    params = {
        "model_type": "logistic_regression",
        "random_seed": 42,
        "class_weight": "balanced",

    }
    metrics = {
        "precision": 0.91,
        "recall": 0.94,
        "pr_auc": 0.97,
    }

    with mlflow.start_run(run_name="tracking_verification") as run:

        # log in mlflow.db
        mlflow.log_params(params=params)
        mlflow.log_metrics(metrics=metrics)

        # log in artifacts folder
        mlflow.log_dict(params, artifact_file="run_evidence/params.json")
        mlflow.log_dict(metrics, artifact_file="run_evidence/metrics.json")

        print(run.info.experiment_id)
        print(run.info.run_name)
        print(run.info.run_id)
        




if __name__ == "__main__":
    main()