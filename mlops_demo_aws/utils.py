"""
This module contains utils shared between different notebooks
"""
import json
import mlflow
import os


def get_deployed_model_stage_for_env(env):
    """
    Get the model version stage under which the latest deployed model version can be found
    for the current environment
    :param env: Current environment
    :return: Model version stage
    """
    # For a registered model version to be served, it needs to be in either the Staging or Production
    # model registry stage
    # (https://docs.databricks.com/applications/machine-learning/manage-model-lifecycle/index.html#transition-a-model-stage).
    # For models in dev and staging, we deploy the model to the "Staging" stage, and in prod we deploy to the
    # "Production" stage
    _MODEL_STAGE_FOR_ENV = {
        "dev": "Staging",
        "staging": "Staging",
        "prod": "Production",
        "test": "Production",
    }
    return _MODEL_STAGE_FOR_ENV[env]


def get_model_name(env: str, test_mode: bool = False):
    """Get the registered model name for the current environment.

    In dev or when running integration tests, we rely on a hardcoded model name.
    Otherwise, e.g. in production jobs, we read the model name from Terraform config-as-code output.

    Args:
        env (str): Current environment
        test_mode (bool, optional): Whether the notebook is running in test mode.. Defaults to False.

    Returns:
        _type_: Registered Model name.
    """
    if env == "dev" or test_mode:
        resource_name_suffix = _get_resource_name_suffix(test_mode)
        return f"model-serving-mlops-model{resource_name_suffix}"
    else:
        # Read ml model name from model_serving_mlops/terraform
        return _get_ml_config_value(env, "model-serving-mlops_model_name")