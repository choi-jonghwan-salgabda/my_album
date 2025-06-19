def test_key_error_instances():
    from your_module import DatasetMapper  # Replace with the actual import
    import pytest

    dataset_mapper = DatasetMapper()

    with pytest.raises(KeyError) as excinfo:
        dataset_mapper.process_data({"data": "sample_data"})  # Simulate input that causes KeyError

    assert str(excinfo.value) == "'instances'"