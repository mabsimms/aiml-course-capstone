import os
import random

from pathlib import Path
from locust import HttpUser, task, constant

from capstone.cli import load_raw_data
from capstone.dataset import prepare_experiment

_BATCH_SIZE = int(os.environ("LOADTEST_BATCH_SIZE"), 1)
_SAMPLE_POOL_SIZE = int(os.environ.get("LOADTEST_SAMPLE_POOL_SIZE"), 500)

def _load_messages() -> list[dict]:
    directory = Path(os.environ["KAGGLE_DIRECTORY"])
    raw = load_raw_data(None, directory)
    _, df_test, _ = prepare_experiment(raw, stratify_by_source=True)
    sample = df_test.sample(n=min(_SAMPLE_POOL_SIZE, len(df_test)), random_state=42)

_MESSAGES = _load_messages()

class ScoringUser(HttpUser):
    wait_time = constant(0)

    @task
    def score(self):
        payload = {
            "messages": random.sample(_MESSAGES, k=_BATCH_SIZE)
        }
        self.client.post("/score", json=payload)

    