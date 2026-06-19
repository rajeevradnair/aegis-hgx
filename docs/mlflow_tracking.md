MLFLOW EXPERIMENT TRACKING
==========================

Purpose
-------

MLflow records every model-training execution as a traceable experiment run.

Each run connects its configuration, parameters, evaluation metrics, artifacts,
and trained model through a unique run identifier.


Local Storage
-------------

All generated MLflow state is stored beneath:

mlflow/

The SQLite database stores searchable metadata:

- experiments
- runs
- parameters
- scalar metrics
- tags
- artifact locations

The artifact store contains complete run evidence:

- parameters JSON
- metrics JSON
- training configuration
- metrics report
- Joblib model
- MLflow-format model

Parameters and metrics are deliberately stored in both locations.

The database representation supports searching, filtering, sorting, and
comparing runs. The JSON artifacts provide portable evidence that remains
inside each run's artifact package.


Run Lifecycle
-------------

1. Configure the tracking URI.
2. Select or create the experiment.
3. Start one MLflow run.
4. Load and split the dataset.
5. Build and train the model.
6. Evaluate the model.
7. Log parameters and scalar metrics to the tracking database.
8. Log parameter and metric JSON copies to artifact storage.
9. Log the configuration, reports, and trained models.
10. Close the run.

One complete training execution corresponds to one MLflow run.


Local UI
--------

Run from the project root:

mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --port 5000

Then open:

http://127.0.0.1:5000


Storage Mental Model
--------------------

mlflow/mlflow.db

Stores the searchable catalog:

- experiment and run records
- parameters
- scalar metrics
- tags
- status
- artifact locations

mlflow/mlruns/

Stores the complete file-based evidence:

- parameters.json
- metrics.json
- baseline_logistic.yaml
- metrics reports
- Joblib model
- MLflow-format model
- dependency metadata


Limitations
-----------

The current tracking store is local and intended for development.

A shared production environment should use:

- a centralized MLflow tracking server
- a durable metadata database such as PostgreSQL
- object storage such as Amazon S3 for artifacts

