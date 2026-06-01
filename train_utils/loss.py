import torch


def SoftIoULoss(pred, target):
    pred = torch.sigmoid(pred)
    smooth = 0.00

    intersection = pred * target

    intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
    pred_sum = torch.sum(pred, dim=(1, 2, 3))
    target_sum = torch.sum(target, dim=(1, 2, 3))
    loss = (intersection_sum + smooth) / \
           (pred_sum + target_sum - intersection_sum + smooth)

    loss = 1 - torch.mean(loss)
    return loss
