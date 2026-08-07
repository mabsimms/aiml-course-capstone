import time
import os
import platform
import tensorflow as tf
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