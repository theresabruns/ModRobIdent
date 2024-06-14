import torch
import torch.nn as nn


class WassersteinLoss(nn.Module):
    def __init__(self):
        super(WassersteinLoss, self).__init__()

    @staticmethod
    def forward(y_pred, y_true):
        # TODO: include gradient penalty

        batch_size = y_true.shape[0]
        diff = torch.abs(y_pred - y_true)
        diff_flat = diff.view(batch_size, -1)
        # norm_diff = F.normalize(diff_flat, dim=1)
        loss = torch.mean(diff_flat)  # norm_diff

        """
        batch_size = y_true.shape[0]

        # Normalize the tensors to represent empirical distributions
        y_true_norm = F.normalize(y_true, p=1, dim=1)
        y_pred_norm = F.normalize(y_pred, p=1, dim=1)

        # Flatten the tensors
        y_true_flat = y_true_norm.view(batch_size, -1)
        y_pred_flat = y_pred_norm.view(batch_size, -1)

        # Convert the tensors to numpy arrays
        y_true_np = y_true_flat.detach().cpu().numpy()
        y_pred_np = y_pred_flat.detach().cpu().numpy()

        emd = torch.tensor([wasserstein_distance(y_true_np[i], y_pred_np[i]) for i in range(batch_size)],
                           dtype=torch.float32, requires_grad=True)
        loss = torch.mean(emd)
        """
        return loss


class CombinedLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(CombinedLoss, self).__init__()
        self.wasserstein = WassersteinLoss()
        self.bce = nn.BCELoss()
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        wasserstein_loss = self.wasserstein(y_pred, y_true)
        bce_loss = self.bce(y_pred, y_true)
        return (1 - self.gamma) * wasserstein_loss + self.gamma * bce_loss


class JaccardLoss(nn.Module):
    def __init__(self):
        super(JaccardLoss, self).__init__()

    @staticmethod
    def forward(labels, outputs):
        label_set = set(labels)
        output_set = set(outputs)
        intersection = label_set.intersection(output_set)
        union = label_set.union(output_set)
        iou = len(intersection) / len(union)
        return iou
