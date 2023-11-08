import torch
import torch.nn as nn
import torch.nn.functional as F

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # 调整 target 和 mask 以匹配 input 的维度
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]

        if input.dim() == 3:
            # 三维情况（训练阶段）
            target = target.unsqueeze(-1)  # 在最后一个维度扩展 target
            output = -input.gather(2, target.long()).squeeze(2) * mask
        elif input.dim() == 2:
            # 二维情况（评估阶段）
            if mask.size(1) != input.size(1):
                if mask.size(1) < input.size(1):
                    padding_size = input.size(1) - mask.size(1)
                    mask = F.pad(mask, (0, padding_size), 'constant', 0)
                else:
                    mask = mask[:, :input.size(1)]

            output = -input * mask

        output = torch.sum(output) / (torch.sum(mask) + 1e-9)  # 为防止除以零的情况
        return output

def compute_loss(output, reports_ids, reports_masks):
    criterion = LanguageModelCriterion()
    loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
    return loss
