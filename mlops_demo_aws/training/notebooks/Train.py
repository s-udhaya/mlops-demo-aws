# Databricks notebook source
##################################################################################
# Model Training Notebook
##
# This notebook runs the MLflow Regression Recipe to train and registers an MLflow model in the model registry.
# It is configured and can be executed as the "Train" task in the model_training_job workflow defined under
# ``mlops_demo_aws/databricks-resources/model-workflow-resource.yml``
#
# NOTE: In general, we recommend that you do not modify this notebook directly, and instead update data-loading
# and model training logic in Python modules under the `steps` directory.
# Modifying this notebook can break model training CI/CD.
#
# However, if you do need to make changes (e.g. to remove the use of MLflow Recipes APIs),
# be sure to preserve the following interface expected by CI and the production model training job:
#
# Parameters:
#
# * env (optional): Name of the environment the notebook is run in (test, staging, or prod).
#                   You can add environment-specific logic to this notebook based on the value of this parameter,
#                   e.g. read training data from different tables or data sources across environments.
#                   The `databricks-dev` profile will be used if users manually run the notebook.
#                   The `databricks-test` profile will be used during CI test runs.
#                   This separates the potentially many runs/models logged during integration tests from
#                   runs/models produced by staging/production model training jobs. You can use the value of this
#                   parameter to further customize the behavior of this notebook based on whether it's running as
#                   a test or for recurring model training in staging/production.
#
#
# Return values:
# * model_uri: The notebook must return the URI of the registered model as notebook output specified through
#              dbutils.notebook.exit() AND as a task value with key "model_uri" specified through
#              dbutils.jobs.taskValues(...), for use by downstream notebooks.
#
# For details on MLflow Recipes and the individual split, transform, train etc steps below, including usage examples,
# see the Regression Recipe overview documentation: https://mlflow.org/docs/latest/recipes.html
# and Regression Recipes API documentation: https://mlflow.org/docs/latest/python_api/mlflow.recipes.html
##################################################################################

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import os
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path
%cd ../

# COMMAND ----------


env = dbutils.widgets.get("env")
experiment_name = dbutils.widgets.get("experiment_name")
model_name = dbutils.widgets.get("model_name")

# COMMAND ----------

import mlflow
import sklearn

from datetime import timedelta, datetime
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F, types as T
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

client = MlflowClient()

# COMMAND ----------

mlflow.set_experiment(experiment_name)

# COMMAND ----------

# Read data and add a unique id column (not mandatory but preferred)
raw_df = (spark.read.format("parquet")
          .load("/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet/")
         )

display(raw_df)

# COMMAND ----------

features_list = ["bedrooms", "neighbourhood_cleansed", "accommodates", "cancellation_policy", "beds", "host_is_superhost", "property_type", "minimum_nights", "bathrooms", "host_total_listings_count", "number_of_reviews", "review_scores_value", "review_scores_cleanliness"]


LABEL_COL = "price" # Name of ground-truth labels column


train_df, baseline_test_df, inference_df = raw_df.select(*features_list+[LABEL_COL]).randomSplit(weights=[0.6, 0.2, 0.2], seed=42)

# COMMAND ----------

# Define the training datasets
X_train = train_df.drop(LABEL_COL).toPandas()
Y_train = train_df.select(LABEL_COL).toPandas().values.ravel()

# Define categorical preprocessor
categorical_cols = [col for col in X_train if X_train[col].dtype == "object"]
one_hot_pipeline = Pipeline(steps=[("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer([("onehot", one_hot_pipeline, categorical_cols)], remainder="passthrough", sparse_threshold=0)

# Define the model
skrf_regressor = RandomForestRegressor(
  bootstrap=True,
  criterion="squared_error",
  max_depth=5,
  max_features=0.5,
  min_samples_leaf=0.1,
  min_samples_split=0.15,
  n_estimators=36,
  random_state=42,
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", skrf_regressor),
])

# Enable automatic logging of input samples, metrics, parameters, and models
mlflow.sklearn.autolog(log_input_examples=True, silent=True)

with mlflow.start_run(run_name="random_forest_regressor") as mlflow_run:
    model.fit(X_train, Y_train)

# COMMAND ----------

# Register model to MLflow Model Registry
run_id = mlflow_run.info.run_id
model_version = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=model_name)

# COMMAND ----------

model_uri = f"models:/{model_version.name}/{model_version.version}"
dbutils.jobs.taskValues.set("model_uri", model_uri)
dbutils.jobs.taskValues.set("model_name", model_version.name)
dbutils.jobs.taskValues.set("model_version", model_version.version)
dbutils.notebook.exit(model_uri)
