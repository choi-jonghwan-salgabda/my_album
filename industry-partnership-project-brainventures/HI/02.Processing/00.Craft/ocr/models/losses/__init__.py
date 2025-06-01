from .CRAFTLoss import CRAFTLoss
from hydra.utils import instantiate


def get_loss_by_cfg(config):
    loss = instantiate(config)
    return loss
