# Databricks notebook source
##################################################################################
# Helper notebook to set up a Databricks Model Serving endpoint.
# This notebook is run after the TriggerModelDeploy.py notebook as part of a multi-task job,
# in order to set up a Model Serving endpoint following the deployment step.
#
#
# This notebook has the following parameters:
#
#  * env (required)  - String name of the current environment for model deployment
#                      (staging, or prod)
# * test_mode (optional): Whether the current notebook is running in "test" mode. Defaults to False. In the provided
#                         notebook, when test_mode is True, an integration test for the model serving endpoint is run.
#                         If false, deploy model serving endpoint.
#  * model_name (required)  - The name of the model in Databricks Model Registry to be served.
#                            This parameter is read as a task value
#                            (https://learn.microsoft.com/azure/databricks/dev-tools/databricks-utils#get-command-dbutilsjobstaskvaluesget),
#                            rather than as a notebook widget. That is, we assume a preceding task (the Train.py
#                            notebook) has set a task value with key "model_name".
#  * model_version (required) - The version of the model in Databricks Model Registry to be served.
#                            This parameter is read as a task value
#                            (https://learn.microsoft.com/azure/databricks/dev-tools/databricks-utils#get-command-dbutilsjobstaskvaluesget),
#                            rather than as a notebook widget. That is, we assume a preceding task (the Train.py
#                            notebook) has set a task value with key "model_version".
##################################################################################


# List of input args needed to run the notebook as a job.
# Provide them via DB widgets or notebook arguments.
#
# Name of the current environment

dbutils.widgets.dropdown("env", "None", ["None", "test", "staging", "prod"], "Environment Name")
# Test mode
dbutils.widgets.dropdown("test_mode", "False", ["True", "False"], "Test Mode")


# COMMAND ----------

# import os
# import sys
# notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
# %cd $notebook_path
# %cd ..
# sys.path.append("../..")

# COMMAND ----------


# get API URL and token
databricks_url = dbutils.secrets.get(scope = "udhay-mlops-demo", key = "db_host_mlops")
databricks_token = dbutils.secrets.get(scope = "udhay-mlops-demo", key = "db_token_mlops")


# COMMAND ----------
import sys

sys.path.append("..")

# COMMAND ----------
env = dbutils.widgets.get("env")
_test_mode = dbutils.widgets.get("test_mode")
test_mode = True if _test_mode.lower() == "true" else False
model_name = dbutils.jobs.taskValues.get("Train", "model_name", debugValue="test-mlops-demo-aws-model")
model_version = dbutils.jobs.taskValues.get("Train", "model_version", debugValue="1")
assert env != "None", "env notebook parameter must be specified"
assert model_name != "", "model_name notebook parameter must be specified"
assert model_version != "", "model_version notebook parameter must be specified"


# COMMAND ----------
from mlops_demo_aws.deployment.model_serving.serving import perform_integration_test, perform_prod_deployment
from mlops_demo_aws.utils import get_deployed_model_stage_for_env, get_model_name

model_name = get_model_name(env)
endpoint_name = f"{model_name}-{env}"
strage = get_deployed_model_stage_for_env(env)

github_repo = dbutils.secrets.get("udhay-mlops-demo", "github_repo")
token = dbutils.secrets.get("udhay-mlops-demo", "github_token")

github_server = "https://api.github.com"


cd_trigger_url = f"{github_server}/repos/{github_repo}/actions/workflows/mlops-demo-aws-deploy-model-serving-{env}.yml/dispatches"
authorization = f"token {token}"

# COMMAND ----------
import requests

response = requests.post(
    cd_trigger_url,
    json={"ref": "main", "inputs": {"modelUri": model_name}},
    headers={"Authorization": authorization},
)
assert response.ok, (
    f"Triggering CD workflow {cd_trigger_url} for model {model_name} "
    f"failed with status code {response.status_code}. Response body:\n{response.content}"
)

# COMMAND ----------
print(
    f"Successfully triggered model CD deployment workflow for {model_name}. See your CD provider to check the "
    f"status of model deployment"
)

# if test_mode:
#     endpoint_name = f"{model_name}-integration-test-endpoint"
#     perform_integration_test(endpoint_name, model_name, model_version, latency_p95_threshold=1000, qps_threshold=1)
#
# elif not test_mode:
#     endpoint_name = f"{model_name}-v{model_version}"
#     perform_prod_deployment(endpoint_name, model_name, model_version, latency_p95_threshold=1000, qps_threshold=1)
#
#

