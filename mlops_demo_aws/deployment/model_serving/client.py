import json
from typing import List, Dict
import requests
from endpoint import Endpoint


class EndpointClient:
    """
    Wrapper class around Databricks Serverless Realtime
    Inference Endpoints.
    """

    def __init__(self, base_url: str, token: str):
        """
        Instantiates an EndpointClient. Parameters.

        base_url: A string pointing to your workspace URL.
        token: Access Token for interacting with your Databricks Workspace.
        """
        self.base_url = base_url
        self.token = token
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def create_inference_endpoint(
        self, data: Dict = None
    ):
        return self._post(uri=Endpoint.SERVING.value, body=data)

    def get_inference_endpoint(self, endpoint_name: str) -> Dict:
        """
        Gets info on the inference endpoint.

        endpoint_name: Serving endpoint name.
        """

        return self._get(f"{Endpoint.SERVING.value}/{endpoint_name}")

    def list_inference_endpoints(self) -> Dict:
        """
        Lists all running inference endpoints.
        """

        return self._get(Endpoint.SERVING.value)

    def update_served_models(
            self, data: Dict = None
    ):
        return self._put(Endpoint.CONFIG.value.format(data["name"]), data)

    def delete_inference_endpoint(self, endpoint_name: str) -> Dict:
        """
        Deletes an inference endpoint.

        endpoint_name: Serving endpoint name.
        """

        return self._delete(f"{Endpoint.SERVING.value}/{endpoint_name}")

    def query_inference_endpoint(self, endpoint_name: str, data: Dict) -> Dict:
        """
        Makes HTTP requests to an inference endpoint.

        endpoint_name: Serving endpoint name.
        data: Payload containing the data expected by the model.
        """

        return self._post(Endpoint.INVOCATIONS.value.format(endpoint_name), data)

    # Debugging

    def get_served_model_build_logs(
        self, endpoint_name: str, served_model_name: str
    ) -> Dict:
        """
        Gets the build logs for the specified endpoint/model.

        endpoint_name: Serving endpoint name.
        served_model_name: Served model name.
        """

        served_models_path = Endpoint.SERVED_MODELS.value.format(endpoint_name)
        build_logs_path = f"{served_model_name}/build-logs"
        return self._get(f"{served_models_path}/{build_logs_path}")

    def get_served_model_server_logs(
        self, endpoint_name: str, served_model_name: str
    ) -> Dict:
        """
        Gets the server logs for the specified endpoint/model.

        endpoint_name: Serving endpoint name.
        served_model_name: Served model name.
        """

        served_models_path = Endpoint.SERVED_MODELS.value.format(endpoint_name)
        server_logs_path = f"{served_model_name}/build-logs"
        return self._get(f"{served_models_path}/{server_logs_path}")

    def get_inference_endpoint_events(self, endpoint_name: str) -> Dict:
        """
        Gets the build endpoint events for the specified endpoint.

        endpoint_name: Serving endpoint name.
        """
        return self._get(Endpoint.EVENTS.value.format(endpoint_name))

    # REST API methods
    def _get(self, uri) -> Dict:
        url = f"{self.base_url}/{uri}"
        response = requests.request("GET", url=url, headers=self.headers)
        return self._handle_api_error(response)

    def _post(self, uri, body) -> Dict:
        json_body = json.dumps(body)
        url = f"{self.base_url}/{uri}"
        response = requests.request(
            "POST", url=url, headers=self.headers, data=json_body
        )
        return self._handle_api_error(response)

    def _put(self, uri, body) -> Dict:
        json_body = json.dumps(body)
        url = f"{self.base_url}/{uri}"
        response = requests.request(
            "PUT", url=url, headers=self.headers, data=json_body
        )
        return self._handle_api_error(response)

    def _delete(self, uri) -> Dict:
        url = f"{self.base_url}/{uri}"
        response = requests.request("DELETE", url=url, headers=self.headers)
        return self._handle_api_error(response)

    def _handle_api_error(self, response):
        if response.status_code == requests.codes.ok:
            return response.json()
        elif response.status_code == requests.codes.bad_request:
            raise Exception(f"Bad request: {response.text}")
        elif response.status_code == requests.codes.unauthorized:
            raise Exception(f"Unauthorized: {response.text}")
        elif response.status_code == requests.codes.not_found:
            raise Exception(f"Not found: {response.text}")
        elif response.status_code == requests.codes.internal_server_error:
            raise Exception(f"Internal server error: {response.text}")
        else:
            raise Exception(f"Unhandled error: {response.text}")
