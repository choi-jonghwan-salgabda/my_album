from .CRAFTHead import CRAFTHead, CRAFTPostProcessor
from hydra.utils import instantiate

def get_head_by_cfg(config):
    head = instantiate(config)
    return head