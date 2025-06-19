cd ~/SambaData/OwnerData/source_backup/my_utils
 
cp -r * ~/SambaData/Backup/FastCamp/Myproject/my_utils/
cd  ~/SambaData/Backup/FastCamp/Myproject/my_utils/
poetry lock
poetry add PyYAML
poetry install
poetry run python -m my_utils.config_utils.configger
ls
poetry run python -m my_utils.config_utils.SimpleLogger
poetry run python -m my_utils.config_utils.SimpleLogger
cp src/my_utils/config_utils/SimpleLogger.py ~/SambaData/OwnerData/source_backup/my_utils/src/my_utils/config_utils/
poetry run python -m my_utils.config_utils.configger
ls
