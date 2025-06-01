from .VGGBackbone import VGGBackbone
from hydra.utils import instantiate


def get_encoder_by_cfg(config):
    encoder = instantiate(config)
    return encoder
