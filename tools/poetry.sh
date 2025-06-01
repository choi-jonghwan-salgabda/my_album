  curl -sSL https://install.python-poetry.org | python3 -
  poetry --version
  poetry env info
  poetry init
  poetry install
  poetry shell
  poetry env activate
  poetry env info --path
  poetry self add poetry-plugin-shell
  poetry env info
  poetry env activate
  poetry --version
  poetry init # 이름, 버전 등 입력 (기본값 사용 가능)
  source $(poetry env info --path)/bin/activate
  poetry env info
  poetry shell
  . /data/ephemeral/home/.cache/pypoetry/virtualenvs/face-finder-project-pRoLg7uv-py3.10/bin/activate
  poetry install
  poetry shell
