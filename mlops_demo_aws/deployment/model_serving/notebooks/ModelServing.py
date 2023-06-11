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


# COMMAND ----------

env = dbutils.widgets.get("env")
model_name = dbutils.jobs.taskValues.get("Train", "model_name", debugValue="test-mlops-demo-aws-model")
model_version = dbutils.jobs.taskValues.get("Train", "model_version", debugValue="1")
assert env != "None", "env notebook parameter must be specified"
assert model_name != "", "model_name notebook parameter must be specified"
assert model_version != "", "model_version notebook parameter must be specified"

# COMMAND ----------

if env == "test":
    mode = "integration_test"
else:
    mode = "production_deployment"

# COMMAND ----------

import random
import string
import datetime
import requests
import time

github_repo = dbutils.secrets.get("udhay-mlops-demo", "github_repo")
github_owner = dbutils.secrets.get("udhay-mlops-demo", "github_owner")
github_token = dbutils.secrets.get("udhay-mlops-demo", "github_token")
workflow = f"mlops-demo-aws-deploy-model-serving-dispatch-{env}.yml"

authorization = f"Token {github_token}"

github_server = "https://api.github.com"

# generate a random id
run_identifier = ''.join(random.choices(string.ascii_uppercase + string.digits, k=15))
# filter runs that were created after this date minus 5 minutes
delta_time = datetime.timedelta(minutes=5)
run_date_filter = (datetime.datetime.utcnow() - delta_time).strftime("%Y-%m-%dT%H:%M")

cd_trigger_url = f"{github_server}/repos/{github_owner}/{github_repo}/actions/workflows/{workflow}/dispatches"
workflow_id = ""

response = requests.post(
    cd_trigger_url,
    json={"ref": "main", "inputs": {"id": run_identifier, "modelName": model_name, "modelVersion": model_version,
                                    "deploymentMode": mode}},
    headers={"Authorization": authorization},
)

print(f"dispatch workflow status: {response.status_code} | workflow identifier: {run_identifier}")

while workflow_id == "":
    get_run_url = f"{github_server}/repos/{github_owner}/{github_repo}/actions/runs?created=%3E{run_date_filter}"
    r = requests.get(get_run_url,
                     headers={"Authorization": authorization})
    runs = r.json()["workflow_runs"]

    if len(runs) > 0:
        for workflow in runs:
            jobs_url = workflow["jobs_url"]
            print(f"get jobs_url {jobs_url}")

            r = requests.get(jobs_url, headers={"Authorization": authorization})

            jobs = r.json()["jobs"]
            if len(jobs) > 0:
                # we only take the first job, edit this if you need multiple jobs
                job = jobs[0]
                steps = job["steps"]
                if len(steps) >= 2:
                    second_step = steps[1]  # if you have position the run_identifier step at 1st position
                    if second_step["name"] == run_identifier:
                        workflow_id = job["run_id"]
                else:
                    print("waiting for steps to be executed...")
                    time.sleep(3)
            else:
                print("waiting for jobs to popup...")
                time.sleep(3)
    else:
        print("waiting for workflows to popup...")
        time.sleep(3)

print(f"workflow_id: {workflow_id}")

# COMMAND ----------

import time

get_job_url = f"{github_server}/repos/{github_owner}/{github_repo}/actions/runs/{workflow_id}/jobs"
r = requests.get(get_job_url,
                 headers={"Authorization": authorization})
sleep_interval = 10
while r.json()["jobs"][0]["conclusion"] is None:
    time.sleep(sleep_interval)
    r = requests.get(get_job_url,
                     headers={"Authorization": authorization})
    print("job is not finished, waiting for {sleep_interval} more seconds")

if r.json()["jobs"][0]["conclusion"] != "success":
    raise ("Model serving is failed")
else:
    print("Model serving is successful")
