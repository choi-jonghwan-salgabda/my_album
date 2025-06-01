poetry env remove python
poetry config virtualenvs.in-project true
poetry env use python3.10
source $(poetry env info --path)/bin/activate
#poetry lock
poetry install
poetry update
poetry update package
poetry run pip install -r requirements.txt
pip install --upgrade pip
