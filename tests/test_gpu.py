import pytest

def test_gpu_available():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    assert len(gpus) > 0, "No GPU found"
