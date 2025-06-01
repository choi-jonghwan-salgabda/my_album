
pip install torchmetrics
pip install Polygon3
pip install shapely
pip install scipy numpy shapely
#평가 방법

source deactivate 

cd /data/ephemeral/home/industry-partnership-project-brainventures

PYTHONPATH=$PYTHONPATH:$(pwd) python Mertric/00_calc_score_cleval.py


