from enum import Enum


class Endpoint(Enum):
    SERVING = "api/2.0/preview/serving-endpoints"
    INVOCATIONS = "realtime-inference/{}/invocations"
    SERVED_MODELS = "api/2.0/preview/serving-endpoints/{}/served-models"
    EVENTS = "api/2.0/preview/serving-endpoints/{}/events"
    CONFIG = "api/2.0/preview/serving-endpoints/{}/config"
