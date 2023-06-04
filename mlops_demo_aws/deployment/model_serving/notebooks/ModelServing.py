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
dbutils.widgets.dropdown("env", "None", ["None", "staging", "prod"], "Environment Name")
# Test mode
dbutils.widgets.dropdown("test_mode", "False", ["True", "False"], "Test Mode")

# COMMAND ----------

import os
import sys
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path
%cd ..
sys.path.append("../..")

# COMMAND ----------

from client import EndpointClient

# get API URL and token
databricks_url = dbutils.secrets.get(scope = "tokens", key = "db_host_mlops")
databricks_token = dbutils.secrets.get(scope = "tokens", key = "db_token_mlops")

# COMMAND ----------

client = EndpointClient(databricks_url, databricks_token)

# COMMAND ----------

# MAGIC %md
# MAGIC ## List existing endpoints

# COMMAND ----------

client.list_inference_endpoints()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Create an endpoint with your model
# MAGIC
# MAGIC In the endpoint object returned by the create call, we can see that our endpoint’s update state is `IN_PROGRESS` and our served model is in a `CREATING` state. The `pending_config` field shows the details of the update in progress.

# COMMAND ----------

env = dbutils.widgets.get("env")
_test_mode = dbutils.widgets.get("test_mode")
test_mode = True if _test_mode.lower() == "true" else False
model_name = dbutils.jobs.taskValues.get("Train", "model_name", debugValue="")
model_version = dbutils.jobs.taskValues.get("Train", "model_version", debugValue="")
print(f"model name: {model_name}")
print(f"model version: {model_version}")
assert env != "None", "env notebook parameter must be specified"
assert model_name != "", "model_name notebook parameter must be specified"
assert model_version != "", "model_version notebook parameter must be specified"

# COMMAND ----------

endpoint_interface = client.get_inference_endpoint(endpoint_name)

# COMMAND ----------

env = "staging"
model_name = "staging-mlops-demo-aws-model"
model_version = "3"
endpoint_name = f"{model_name}-endpoint"

dbfs_table_path = "/Users/udhayaraj.sivalingam@databricks.com/data/inference_tables"

data = {
    "name": endpoint_name,
    "config": {
        "served_models": [
    {
        "model_name": model_name,
        "model_version": model_version,
        "workload_size": "Small",
        "scale_to_zero_enabled": False,
    }
        ]
    },
    "inference_table_config": {
        "dbfs_destination_path": dbfs_table_path
    }
}
try:
    endpoint_interface = client.get_inference_endpoint(endpoint_name)
    print(f"Endpoint {endpoint_name} is being updated")
    client.update_served_models(data)
except:
    print(f"New endpoint {endpoint_name} is being created")
    client.create_inference_endpoint(data)

# COMMAND ----------

# env = "staging"
# model_name = "staging-mlops-demo-aws-model"
# model_version = "3"
endpoint_name = f"{model_name}_version_{model_version}"
models = [
    {
        "model_name": model_name,
        "model_version": model_version,
        "workload_size": "Small",
        "scale_to_zero_enabled": False,
    }
]
client.create_inference_endpoint(endpoint_name, models)

# COMMAND ----------

client.list_inference_endpoints()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Check the state of your endpoint
# MAGIC
# MAGIC We can check on the status of our endpoint to see if it is ready to receive traffic. Note that when the update is complete and the endpoint is ready to be queried, the `pending_config` is no longer populated.

# COMMAND ----------

import time

endpoint = client.get_inference_endpoint(endpoint_name)

while endpoint['state']['config_update'] == "IN_PROGRESS":
  time.sleep(5)
  endpoint = client.get_inference_endpoint(endpoint_name)
  print(endpoint["name"], endpoint["state"])

# COMMAND ----------

client.get_inference_endpoint(endpoint_name)


