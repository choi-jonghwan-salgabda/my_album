poetry env list
poetry env remove --all
whereis python3.10
poetry env use /usr/bin/python3.10
poetry add dlib faiss-cpu scikit-learn opencv-python numpy
poetry env activate
poetry run python
