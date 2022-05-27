import torch
from torch import nn

from structures.image_list import to_image_list

from .backbone import build_backbone
from .head.detector_head import bulid_head

from model.layers.uncert_wrapper import make_multitask_wrapper

class KeypointDetector(nn.Module):
    '''
    Generalized structure for keypoint based object detector.
    main parts:
    - backbone
    - heads
    '''

    def __init__(self, cfg):
        super(KeypointDetector, self).__init__()

        self.backbone = build_backbone(cfg)
        self.heads = bulid_head(cfg, self.backbone.out_channels)
        self.test = cfg.DATASETS.TEST_SPLIT == 'test'

    def forward(self, images, depths=None, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        # if self.training:
        #     self.backbone.testdropout = False

        images = to_image_list(images).tensors
        depths = to_image_list(depths).tensors if depths is not None else None
        features = self.backbone(images, depths)

        if self.training:
            loss_dict, log_loss_dict = self.heads(features, targets)
            return loss_dict, log_loss_dict
        else:
            result, eval_utils, visualize_preds = self.heads(features, targets, test=self.test)
            return result, eval_utils, visualize_preds