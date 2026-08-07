import tensorflow as tf
import os

def configure_gpu(mode: str | None = None) -> list:
    if mode is None:
        mode = os.environ.get("GPU_MODE", "auto")

    if mode == "force_cpu":
        tf.config.set_visible_devices([], "GPU")
        return []

    gpus = tf.config.list_physical_devices("GPU")
    if mode == "force_gpu" and not gpus:
        raise RuntimeError(
            "GPU_MODE=force_gpu but not GPU was detected - refusing to start silent on CPU mode"
        )
    return gpus