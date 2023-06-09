import gevent.monkey

gevent.monkey.patch_all()

import json
import os

from locust import HttpUser, task
from locust.env import Environment
from locust.stats import stats_history, stats_printer
import pandas as pd


class DataContext(object):
    """DataContext."""

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(DataContext, cls).__new__(cls)
        return cls.instance


class TestUser(HttpUser):
    """TestUser."""

    def __init__(self, *args, **kwargs):
        """Initialize."""
        super().__init__(*args, **kwargs)

        self.endpoint_name = None
        self.api = "/invocations"
        self.headers = None

        self.json_payload = None

    def on_start(self):
        self.headers = {
            "Authorization": f'Bearer {os.environ.get("DATABRICKS_TOKEN")}',
            "Content-Type": "application/json",
        }

        df = DataContext().sample_data.copy()
        for col in df.columns:
            if "datetime" in df[col].dtype.name:
                df[col] = df[col].astype(str)
        ds_dict = {"dataframe_split": df.to_dict(orient="split")}
        self.json_payload = json.dumps(ds_dict, allow_nan=True)

        return super().on_start()

    @task
    def invoke_model(self):
        self.client.post(self.api, headers=self.headers, data=self.json_payload)


def test_endpoint_locust(
    endpoint_name: str,
    latency_threshold: int,
    qps_threshold: int,
    test_data_df: pd.DataFrame,
    active_users: int = 1,
    duration: int = 20,
):
    DataContext().sample_data = test_data_df

    db_host = os.environ.get("DATABRICKS_HOST")
    if db_host.endswith("/"):
        db_host = db_host[:-1]

    host = f"{db_host}/serving-endpoints/{endpoint_name}"

    env = Environment(user_classes=[TestUser], host=host)

    runner = env.create_local_runner()

    env.events.init.fire(environment=env, runner=runner)
    gevent.spawn(stats_printer(env.stats))

    gevent.spawn(stats_history, env.runner)

    # spawn rate: 0.5  = 1 user every 2 seconds
    env.runner.start(active_users, spawn_rate=0.5)

    gevent.spawn_later(duration, lambda: runner.quit())

    env.runner.greenlet.join()
    key = list(runner.stats.entries.keys())[0]
    stat_entry = runner.stats.entries[key]
    qps = (1.0 / stat_entry.avg_response_time) * 1000.0  # 1 / ms * ms
    p95 = stat_entry.get_response_time_percentile(0.95)

    if stat_entry.num_failures > 0:
        error_key = list(runner.stats.errors.keys())[0]
        se = runner.stats.errors[error_key]
        raise Exception(f"Request failed with status {se.error}")

    if p95 > latency_threshold:
        raise Exception(
            f"Latency requirements are not met. Observed P99 for the requests is {p95}. At least {latency_threshold} was expected."
        )
    if qps < qps_threshold:
        raise Exception(
            f"QPS requirements are not met. Observed QPS for the requests is {qps}. At least {qps_threshold} was expected."
        )
