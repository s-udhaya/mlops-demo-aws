import gevent.monkey

gevent.monkey.patch_all()

import json
import logging
import os
import pathlib
import time
from typing import Any

import click as click
from databricks_cli.configure.config import _get_api_client
from databricks_cli.configure.provider import EnvironmentVariableConfigProvider
from databricks_cli.sdk import ApiClient
from mlflow.client import MlflowClient
from model_serving_mlops.deployment.model_deployment.endpoint_performance import test_endpoint_locust
from model_serving_mlops.utils import get_deployed_model_stage_for_env, get_model_name
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
import requests
from requests.exceptions import HTTPError
import yaml

PRODUCTION_DEPLOYMENT = "production_deployment"
INTEGRATION_TEST = "integration_test"

logger = logging.getLogger(__name__)
mlflow_client = MlflowClient()


def create_spark_session():
    return SparkSession.builder.master("local[1]").getOrCreate()


def prepare_scoring_data() -> pd.DataFrame:
    input_path = pathlib.Path.cwd() / "model_serving_mlops" / "training" / "data" / "sample.parquet"
    input_pdf = pd.read_parquet(str(input_path.absolute()))
    return input_pdf.drop(columns=["fare_amount"])


def get_model_version_for_stage(model_name: str, stage: str) -> str:
    versions = mlflow_client.get_latest_versions(model_name, stages=[stage])
    return str(max([int(v.version) for v in versions]))


def get_recent_model_version(name: str) -> int:
    model_versions_list = mlflow_client.get_latest_versions(name)
    return max(int(mv.version) for mv in model_versions_list)


def get_api_clent() -> ApiClient:
    logger.info("Getting config using EnvironmentVariableConfigProvider...")
    config = EnvironmentVariableConfigProvider().get_config()
    logger.info(f"config: {config}")
    logger.info("_get_api_client")
    api_client = _get_api_client(config, command_name="")
    return api_client


def read_config(file_name: str) -> dict[str, str]:
    file_content = (pathlib.Path("conf") / file_name).read_text()
    return json.loads(file_content)


def create_serving_endpoint(api_client: ApiClient, endpoint_name: str, model_name: str, model_version: str):
    req = {
        "name": endpoint_name,
        "config": {
            "served_models": [
                {
                    "name": model_name,
                    "model_name": model_name,
                    "model_version": model_version,
                    "workload_size": "Small",
                    "scale_to_zero_enabled": False,
                }
            ],
            "traffic_config": {"routes": [{"served_model_name": model_name, "traffic_percentage": 100}]},
        },
    }
    return api_client.perform_query("POST", "/serving-endpoints", data=req)


def delete_endpoint(api_client: ApiClient, endpoint_name: str):
    try:
        return api_client.perform_query("DELETE", f"/serving-endpoints/{endpoint_name}")
    except Exception as e:
        return str(e)


def check_if_endpoint_is_ready(api_client: ApiClient, endpoint_name: str):
    res = api_client.perform_query("GET", f"/serving-endpoints/{endpoint_name}")
    print(res)
    state = res.get("state")
    if state:
        return state.get("ready") == "READY"
    else:
        return False


def wait_for_endpoint_to_become_ready(
    api_client: ApiClient, endpoint_name: str, timeout: int = 360, step: int = 20
) -> bool:
    waited = 0
    while not check_if_endpoint_is_ready(api_client, endpoint_name):
        waited += step
        if waited >= timeout:
            return False
        time.sleep(step)
    return True


def query_endpoint(endpoint_name: str, df: pd.DataFrame) -> tuple[Any, int]:
    http_path = pathlib.Path(os.environ.get("DATABRICKS_HOST")) / "serving-endpoints" / endpoint_name / "invocations"
    http_url = str(http_path).replace("https:/", "https://")
    headers = {
        "Authorization": f'Bearer {os.environ.get("DATABRICKS_TOKEN")}',
        "Content-Type": "application/json",
    }
    df = df.copy()
    for col in df.columns:
        if "datetime" in df[col].dtype.name:
            df[col] = df[col].astype(str)
    ds_dict = {"dataframe_split": df.to_dict(orient="split")}
    data_json = json.dumps(ds_dict, allow_nan=True)
    start_time = time.time_ns()

    response = requests.request(method="POST", headers=headers, url=http_url, data=data_json)
    end_time = time.time_ns()
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}, {response.text}")
    return response.json(), (end_time - start_time) / 1_000_000


def test_endpoint(
    endpoint_name: str,
    latency_threshold: int,
    qps_threshold: int,
    test_data_df: pd.DataFrame,
):
    durations = []
    for _ in range(500):
        res_json, duration = query_endpoint(endpoint_name, test_data_df)
        durations.append(duration)
        preds = res_json.get("predictions")
        if preds:
            if len(preds) != 10:
                raise Exception("Wrong number of predictions!")
    p95 = np.percentile(durations, 95)
    qps = len(durations) / (sum(durations) / 1000)
    print(f"Observed latency (P90): {p95}. Observed QPS: {qps}.")
    if p95 > latency_threshold:
        raise Exception(
            f"Latency requirements are not met. Observed P99 for the requests is {p95}. At least {latency_threshold} was expected."
        )
    if qps < qps_threshold:
        raise Exception(
            f"QPS requirements are not met. Observed QPS for the requests is {qps}. At least {qps_threshold} was expected."
        )


def get_max_version_for_model(api_client: ApiClient, endpoint_name: str, model_name: str):
    res = api_client.perform_query("GET", f"/serving-endpoints/{endpoint_name}")
    if res.status_code == 404:
        return None
    if res.status_code != 200:
        raise Exception(f"Error requesting endpoint! Response code: {res.status_code}")
    served_models = res.json()["config"]["served_models"]
    versions = []
    for model in served_models:
        if model["model_name"] == model_name:
            versions.append(model["model_version"])
    max_version = str(max(int(x) for x in versions))
    return max_version


def deploy_new_version_to_existing_endpoint(
    api_client: ApiClient, endpoint_name: str, model_name: str, model_version: str
):
    update_endpoint_req = {
        "served_models": [
            {
                "name": model_name,
                "model_name": model_name,
                "model_version": model_version,
                "workload_size": "Small",
                "scale_to_zero_enabled": False,
            }
        ],
        "traffic_config": {"routes": [{"served_model_name": model_name, "traffic_percentage": 100}]},
    }
    res = api_client.perform_query("PUT", f"/serving-endpoints/{endpoint_name}/config", data=update_endpoint_req)
    print(res)


def get_model_endpoint_config(api_client: ApiClient, endpoint_name: str) -> dict[str, Any]:
    try:
        res = api_client.perform_query("GET", f"/serving-endpoints/{endpoint_name}")
        return res
    except HTTPError:
        return None


# def deploy_model_serving_endpoint(endpoint_name: str, model_name: str, model_version: int):
#     api_client = get_api_clent()
#     df = prepare_scoring_data[:10]
#     existing_endpoint_conf = get_model_endpoint_config(api_client, endpoint_name)

#     if existing_endpoint_conf:
#         deploy_new_version_to_existing_endpoint(api_client, endpoint_name, model_name, model_version)
#     else:
#         create_serving_endpoint(api_client, endpoint_name, model_name, model_version)
#     time.sleep(100)
#     test_endpoint(endpoint_name, 1000, 1, df)


def perform_integration_test(
    endpoint_name: str, model_name: str, stage: str, latency_p95_threshold: int, qps_threshold: int
):
    logger.info("Getting api_client...")
    api_client = get_api_clent()
    model_version = get_model_version_for_stage(model_name, stage)
    test_data_df = prepare_scoring_data()[:10]
    delete_endpoint(api_client, endpoint_name)
    create_serving_endpoint(api_client, endpoint_name, model_name, model_version)
    time.sleep(100)
    if wait_for_endpoint_to_become_ready(api_client, endpoint_name):
        test_endpoint_locust(endpoint_name, latency_p95_threshold, qps_threshold, test_data_df)
        delete_endpoint(api_client, endpoint_name)
    else:
        print("Endpoint failed to become ready within timeout. ")
        raise Exception("Endpoint failed to become ready within timeout. ")


def perform_prod_deployment(
    endpoint_name: str, model_name: str, stage: str, latency_p95_threshold: int, qps_threshold: int
):
    api_client = get_api_clent()
    df = prepare_scoring_data()[:10]
    existing_endpt_conf = get_model_endpoint_config(api_client, endpoint_name)
    model_version = get_model_version_for_stage(model_name, stage)
    if existing_endpt_conf:
        deploy_new_version_to_existing_endpoint(api_client, endpoint_name, model_name, model_version)
    else:
        create_serving_endpoint(api_client, endpoint_name, model_name, model_version)
    time.sleep(100)
    if wait_for_endpoint_to_become_ready(api_client, endpoint_name):
        test_endpoint_locust(endpoint_name, latency_p95_threshold, qps_threshold, df)
    else:
        raise Exception(f"Production endpoint {endpoint_name} is not ready!")


@click.command()
@click.option(
    "--mode",
    type=click.Choice([INTEGRATION_TEST, PRODUCTION_DEPLOYMENT]),
    default=INTEGRATION_TEST,
    help="""Run mode. Valid values are 'integration_test' for the test deployment in Staging environment
        and  'production_deployment' for model deployment in Production environment""",
)
@click.option(
    "--env",
    type=click.STRING,
    default="staging",
    help="""Target environment. Valid values are "dev", "staging", "prod".""",
)
@click.option(
    "--config",
    required=True,
    type=click.STRING,
    help="""Path to the configuration file.""",
)
def main(mode: str, env: str, config: str):
    model_name = get_model_name(env)
    endpoint_name = f"{model_name}-{env}"
    strage = get_deployed_model_stage_for_env(env)
    with open(config, "r") as file:
        config_dict = yaml.safe_load(file)
    if mode == INTEGRATION_TEST:
        perform_integration_test(
            endpoint_name,
            model_name,
            strage,
            config_dict.get("latency_p95_threshold", 1000),
            config_dict.get("qps_threshold", 1),
        )
    elif mode == PRODUCTION_DEPLOYMENT:
        perform_prod_deployment(
            endpoint_name,
            model_name,
            strage,
            config_dict.get("latency_p95_threshold", 1000),
            config_dict.get("qps_threshold", 1),
        )
    else:
        raise Exception(f"Wrong value for mode parameter: {mode}")


if __name__ == "__main__":
    main()
