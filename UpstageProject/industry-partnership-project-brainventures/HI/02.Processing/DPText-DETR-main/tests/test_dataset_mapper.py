def test_key_error_instances():
    from your_module import DatasetMapper  # Replace with the actual import
    import pytest

    dataset_mapper = DatasetMapper()

    with pytest.raises(KeyError) as excinfo:
        dataset_mapper.map_data({"some_key": "some_value"})  # Simulate input without 'instances'
    
    assert str(excinfo.value) == "'instances'"