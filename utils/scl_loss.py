import torch
import torch.nn as nn


class DDI_SCL_Loss(nn.Module):
    def __init__(self, tau1=0.07, tau2=0.2, beta=0.5, eps=1e-12):
        super(DDI_SCL_Loss, self).__init__()
        self.tau1 = tau1
        self.tau2 = tau2
        self.beta = beta
        self.eps = eps

    def _contrastive_loss(self, z, positive_mask, temperature):
        device = z.device
        total = z.size(0)
        logits = torch.matmul(z, z.T) / max(float(temperature), self.eps)
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

        self_mask = torch.eye(total, dtype=torch.bool, device=device)
        valid_mask = ~self_mask
        positive_mask = positive_mask & valid_mask

        exp_logits = torch.exp(logits) * valid_mask.float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(self.eps))

        pos_count = positive_mask.sum(dim=1)
        valid_anchor = pos_count > 0
        if not torch.any(valid_anchor):
            return z.sum() * 0.0

        mean_log_prob = (positive_mask.float() * log_prob).sum(dim=1) / pos_count.clamp_min(1)
        loss = -mean_log_prob[valid_anchor].mean()
        if torch.isnan(loss) or torch.isinf(loss):
            return z.sum() * 0.0
        return loss

    def forward(self, features, labels, subclass_labels):
        if features.dim() == 2:
            features = features.unsqueeze(1)
        if features.dim() != 3:
            raise ValueError("features must be [batch_size, n_views, proj_dim] or [batch_size, proj_dim]")

        batch_size, n_views, dim = features.shape
        z = features.reshape(batch_size * n_views, dim)

        labels = labels.contiguous().view(-1).to(features.device)
        subclass_labels = subclass_labels.contiguous().view(-1).to(features.device)
        if labels.numel() != batch_size or subclass_labels.numel() != batch_size:
            raise ValueError("labels and subclass_labels must match batch size")

        labels_all = labels.repeat_interleave(n_views)
        subclass_all = subclass_labels.repeat_interleave(n_views)

        same_subclass = subclass_all.view(-1, 1).eq(subclass_all.view(1, -1))
        same_label = labels_all.view(-1, 1).eq(labels_all.view(1, -1))
        different_subclass = ~same_subclass

        loss_sub = self._contrastive_loss(z, same_subclass, self.tau1)
        loss_class = self._contrastive_loss(z, same_label & different_subclass, self.tau2)
        return loss_sub + self.beta * loss_class
