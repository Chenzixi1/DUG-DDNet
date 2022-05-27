import torch
from torch import nn
import pdb

from .detector_predictor import make_predictor
from .detector_loss import make_loss_evaluator
from .detector_infer import make_post_processor

class Detect_Head(nn.Module):
    def __init__(self, cfg, in_channels):
        super(Detect_Head, self).__init__()

        self.predictor = make_predictor(cfg, in_channels)
        self.loss_evaluator = make_loss_evaluator(cfg)
        self.post_processor = make_post_processor(cfg)
        self.testdropout = cfg.TEST.DROPOUT

    def forward(self, features, targets=None, test=False):

        if self.training:

            x = self.predictor(features, targets)

            loss_dict, log_loss_dict = self.loss_evaluator(x, targets)
            return loss_dict, log_loss_dict
        else:
            if self.testdropout:
                output_clss = []
                output_regs = []
                for feature in features:
                    x = self.predictor(feature, targets)
                    output_clss.append(x['cls'])
                    output_regs.append(x['reg'])

                output_clss = torch.stack(output_clss, dim=0)
                output_regs = torch.stack(output_regs, dim=0)
                output_cls = torch.mean(output_clss, dim=0)
                output_reg = torch.mean(output_regs, dim=0)

                depth_epistemic_uncertainty = output_regs[:, :, 48, ...] * output_regs[:, :, 48, ...] - output_reg[:, 48, ...] * output_reg[:, 48, ...]
                depth_epistemic_uncertainty = torch.mean(depth_epistemic_uncertainty, dim=0).unsqueeze(0)
                corners_epistemic_uncertainty = output_regs[:, :, 6:26, ...] * output_regs[:, :, 6:26, ...] - output_reg[:, 6:26, ...] * output_reg[:, 6:26, ...]
                corners_epistemic_uncertainty = torch.mean(corners_epistemic_uncertainty, dim=0)
                x = {'cls': output_cls,
                     'reg': output_reg,
                     'depth_epistemic_uncertainty': depth_epistemic_uncertainty,
                     'corners_epistemic_uncertainty': corners_epistemic_uncertainty}
            else:
                x = self.predictor(features, targets)

            result, eval_utils, visualize_preds = self.post_processor(x, targets, test=test, features=features)
            return result, eval_utils, visualize_preds

def bulid_head(cfg, in_channels):
    
    return Detect_Head(cfg, in_channels)