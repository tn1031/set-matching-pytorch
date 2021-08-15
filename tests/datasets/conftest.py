import pytest


def pytest_addoption(parser):
    parser.addoption("--data_dir", action="store", default=None, help="path to input data directory for dataset tests")


@pytest.fixture
def data_dir(request):
    return request.config.getoption("--data_dir")
