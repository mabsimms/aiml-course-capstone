import pytest

def pytest_addoption(parser):
    parser.addoption("--runtrain", action="store_true", default=False, help="Run training tests (these take a while)")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--runtrain"):
        return
    skip_train = pytest.mark.skip(reason="Set --runtrain to run training suite")
    for item in items:
        if "train" in item.keywords:
            item.add_marker(skip_train)