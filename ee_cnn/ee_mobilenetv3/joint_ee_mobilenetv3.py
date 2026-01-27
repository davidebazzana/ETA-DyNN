import numpy as np
import torch
from torch import nn
from torchvision.models import mobilenet_v3_large
from .side_branch import SideBranch


class Joint_EE_MobileNetV3(nn.Module):
    def __init__(self, num_classes=1, exit_points=(3, 6), thresholds=None, calibrators=None, disable_ee:bool=False, inference:bool=False):
        super().__init__()
        base_model = mobilenet_v3_large(weights='DEFAULT')

        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.classifier = base_model.classifier
        self.exit_points = exit_points
        self.thresholds = thresholds
        self.calibrators = calibrators
        self.disable_ee = disable_ee
        self.inference = inference

        self.exits = nn.ModuleList()
        for idx in exit_points:
            out_channels = self.features[idx].block[-1].out_channels
            self.exits.append(SideBranch(out_channels, num_classes))

    def forward(self, x, force_exit:int|None=None):
        out = x
        if not self.disable_ee: results = []

        for idx, layer in enumerate(self.features):
            out = layer(out)
            if not self.disable_ee and idx in self.exit_points:
                exit_idx = self.exit_points.index(idx)
                exit_output = self.exits[exit_idx](out)

                if self.inference:
                    scores = torch.sigmoid(exit_output.squeeze()).cpu()
                    if scores.shape == torch.Size([]):
                        # Prevent "ValueError: y should be a 1d array, got an array of shape () instead." from self.calibrators[exit_idx].predict(scores) when scores is a scalar.
                        scores = [scores]
                    calibrated_scores = self.calibrators[exit_idx].predict(scores)
                    calibrated_scores[calibrated_scores < self.thresholds[exit_idx]["lower_threshold"]] = 0.
                    calibrated_scores[calibrated_scores > self.thresholds[exit_idx]["upper_threshold"]] = 1.
                    calibrated_scores[(calibrated_scores >= self.thresholds[exit_idx]["lower_threshold"]) & (calibrated_scores <= self.thresholds[exit_idx]["upper_threshold"])] = -1.
                    unique, counts = np.unique(calibrated_scores, return_counts=True)
                    batch_stat = dict(zip(unique, counts))
                    batch_res = max(batch_stat, key=batch_stat.get)
                    if batch_res == 0. or batch_res == 1.:
                        return exit_idx, batch_res, batch_stat
                elif force_exit is not None:
                    results.append(exit_output)
                    return results
                else:
                    results.append(exit_output)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        if self.inference:
            scores = torch.sigmoid(out.squeeze()).cpu()
            if scores.shape == torch.Size([]):
                # Prevent "ValueError: y should be a 1d array, got an array of shape () instead." from self.calibrators[exit_idx].predict(scores) when scores is a scalar.
                scores = [scores]
            calibrated_scores = self.calibrators[-1].predict(scores)
            calibrated_scores[calibrated_scores < self.thresholds[-1]["lower_threshold"]] = 0.
            calibrated_scores[calibrated_scores > self.thresholds[-1]["upper_threshold"]] = 1.
            calibrated_scores[(calibrated_scores >= self.thresholds[-1]["lower_threshold"]) & (calibrated_scores <= self.thresholds[-1]["upper_threshold"])] = -1.
            unique, counts = np.unique(calibrated_scores, return_counts=True)
            batch_stat = dict(zip(unique, counts))
            batch_res = max(batch_stat, key=batch_stat.get)
            return 2, batch_res, batch_stat

        if self.disable_ee:
            return out
        else:
            results.append(out)
            return results

criterion = nn.BCEWithLogitsLoss()
        
def multi_exit_loss(outputs, targets, weights=None):
    if weights is None:
        weights = [1.0 / len(outputs)] * len(outputs)

    total_loss = 0.0
    for out, w in zip(outputs, weights):
        # For each exit, compute the loss.
        # out.shape == (64, 1)
        out = torch.squeeze(out)
        loss = criterion(out, targets)
        total_loss += w * loss

    return total_loss
