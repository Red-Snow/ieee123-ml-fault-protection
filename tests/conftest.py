import pytest

from ieee123fp.config import FEEDER_DIR


def pytest_collection_modifyitems(config, items):
    if (FEEDER_DIR / "IEEE123Master.dss").exists():
        return
    skip = pytest.mark.skip(reason="feeder model missing: run  ieee123fp fetch")
    for item in items:
        if "needs_feeder" in item.keywords:
            item.add_marker(skip)


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_feeder: requires the downloaded OpenDSS feeder files")
