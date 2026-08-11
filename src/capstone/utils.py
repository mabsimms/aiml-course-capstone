import time
import os
import platform
from contextlib import contextmanager
import logging
import subprocess
import psutil


logger = logging.getLogger("capstone")

@contextmanager
def log_duration(operation : str):
    timing = {"seconds": None}
    logger.info("Starting %s", operation)
    start = time.perf_counter()

    try:
        yield timing
    finally:
        timing["seconds"] = time.perf_counter() - start
        logger.info("%s completed in %.1f seconds", operation, timing["seconds"])

def get_machine_info() -> dict:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    gpu_names = []
    for gpu in gpus:
        try:
            details = tf.config.experimental.get_device_details(gpu)
            gpu_names.append(details.get("device_name", "unknown"))
        except Exception:
            gpu_names.append("unknown")

    # Try and pull nvidia information if available
    gpu_memory_info = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        total, used = result.stdout.strip().split(",")
        gpu_memory_info = {
            "total_mb": int(total.strip()),
            "used_mb": int(used.strip())
        }
    except:
        pass

    return { 
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "system_memory_gb": round(psutil.virtual_memory().total / (1024 ** 3), 1),
        "tensorflow_version": tf.__version__,
        "gpus": gpu_names,
        "gpu_memory_info": gpu_memory_info
    }    

def build_metrics_summary(metrics: dict, threshold: float = 0.5) -> dict:
    return { 
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "threshold": threshold,
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "classification_report": metrics["classification_report"],
    }

def get_git_info() -> dict | None:
    import git
    try:
        repo = git.Repo(search_parent_directories=True)
        head_commit = repo.head.commit
        return { 
            "commit": head_commit.hexsha,
            "branch": None if repo.head.is_detached else repo.active_branch.name,
            "commit_timestamp": head_commit.committed_datetime.isoformat(),
            "dirty": repo.is_dirty(untracked_files=True)
        }
    except Exception:
        return None