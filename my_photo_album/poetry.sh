 poetry env remove python
 poetry install

 poetry run pip install face_recognition
 poetry run pip install git+https://github.com/ageitgey/face_recognition_models.git
 poetry run pip install setuptools
 poetry run python -c "import face_recognition_models; print(face_recognition_models.__file__)"
