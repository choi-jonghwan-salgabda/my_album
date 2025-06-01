def pytest_configure():
    import pytest
    from unittest.mock import MagicMock

    pytest.mock_data = MagicMock()
    pytest.mock_data.get_instances = lambda: None

def pytest_unconfigure():
    del pytest.mock_data

def pytest_addoption(parser):
    parser.addoption("--mock-data", action="store_true", help="Use mock data for tests")