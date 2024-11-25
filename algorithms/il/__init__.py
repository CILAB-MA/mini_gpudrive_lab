import torch.nn.functional as F

from algorithms.il.model.bc import *
from algorithms.il.loss import *

MODELS = dict(bc=ContFeedForward, late_fusion=LateFusionBCNet,
              attention=LateFusionAttnBCNet, wayformer=None)

LOSS = dict(
    l1=l1_loss, 
    mse=mse_loss,
    twohot=two_hot_loss, 
    gmm=gmm_loss
)