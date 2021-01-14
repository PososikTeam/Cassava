import torch.nn.functional as F
from catalyst.contrib import registry
from torch import nn
import torch

def log_t(u, t):
    """Compute log_t for `u`."""

    if t == 1.0:
        return torch.log(u)
    else:
        return (u ** (1.0 - t) - 1.0) / (1.0 - t)


def exp_t(u, t):
    """Compute exp_t for `u`."""

    if t == 1.0:
        return torch.exp(u)
    else:
        return torch.relu(1.0 + (1.0 - t) * u) ** (1.0 / (1.0 - t))


def compute_normalization_fixed_point(activations, t, num_iters=5):
    """Returns the normalization value for each example (t > 1.0).
    Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature 2 (> 1.0 for tail heaviness).
    num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """

    mu = torch.max(activations, dim=-1).values.view(-1, 1)
    normalized_activations_step_0 = activations - mu

    normalized_activations = normalized_activations_step_0
    i = 0
    while i < num_iters:
        i += 1
        logt_partition = torch.sum(exp_t(normalized_activations, t), dim=-1).view(-1, 1)
        normalized_activations = normalized_activations_step_0 * (logt_partition ** (1.0 - t))

    logt_partition = torch.sum(exp_t(normalized_activations, t), dim=-1).view(-1, 1)

    return -log_t(1.0 / logt_partition, t) + mu


def compute_normalization(activations, t, num_iters=5):
    """Returns the normalization value for each example.
    Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature 2 (< 1.0 for finite support, > 1.0 for tail heaviness).
    num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """

    if t < 1.0:
        return None # not implemented as these values do not occur in the authors experiments...
    else:
        return compute_normalization_fixed_point(activations, t, num_iters)


def tempered_softmax(activations, t, num_iters=5):
    """Tempered softmax function.
    Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature tensor > 0.0.
    num_iters: Number of iterations to run the method.
    Returns:
    A probabilities tensor.
    """

    if t == 1.0:
        normalization_constants = torch.log(torch.sum(torch.exp(activations), dim=-1))
    else:
        normalization_constants = compute_normalization(activations, t, num_iters)

    return exp_t(activations - normalization_constants, t)


def bi_tempered_logistic_loss(activations, labels, t1, t2, label_smoothing=0.0, num_iters=5, num_classes = 5):

    """Bi-Tempered Logistic Loss with custom gradient.
    Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    labels: A tensor with shape and dtype as activations.
    t1: Temperature 1 (< 1.0 for boundedness).
    t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
    label_smoothing: Label smoothing parameter between [0, 1).
    num_iters: Number of iterations to run the method.
    Returns:
    A loss tensor.
    """
    target = F.one_hot(labels, num_classes).float()

    if label_smoothing > 0.0:
        target = (1 - num_classes / (num_classes - 1) * label_smoothing) * target + label_smoothing / (num_classes - 1)

    probabilities = tempered_softmax(activations, t2, num_iters)

    temp1 = (log_t(target + 1e-10, t1) - log_t(probabilities, t1)) * target
    temp2 = (1 / (2 - t1)) * (torch.pow(target, 2 - t1) - torch.pow(probabilities, 2 - t1))
    loss_values = temp1 - temp2

    return torch.sum(torch.sum(loss_values, dim=-1))
   

@registry.Criterion
class TemperedLogLoss(nn.Module):
    def __init__(self, label_smoothing=0.05, t1 = 0.2, t2 = 4, num_iters = 7):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.t1 = t1
        self.t2 = t2
        self.num_iters = num_iters

    def forward(self, input, target):
        return bi_tempered_logistic_loss(input, target, self.t1, self.t2, self.label_smoothing, self.num_iters, input.size(1))
    
@registry.Criterion
class FocalCosineLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, xent=.1):
        super(FocalCosineLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

        self.xent = xent

        self.y = torch.Tensor([1]).cuda()

    def forward(self, input, target, reduction="mean"):
        cosine_loss = F.cosine_embedding_loss(input, F.one_hot(target, num_classes=input.size(-1)), self.y, reduction=reduction)

        cent_loss = F.cross_entropy(F.normalize(input), target, reduce=False)
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * cent_loss

        if reduction == "mean":
            focal_loss = torch.mean(focal_loss)

        return cosine_loss + self.xent * focal_loss


@registry.Criterion
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smooth_factor=0.05):
        super().__init__()
        self.smooth_factor = smooth_factor

    def _smooth_labels(self, num_classes, target):
        # When label smoothing is turned on,
        # KL-divergence between q_{smoothed ground truth prob.}(w)
        # and p_{prob. computed by model}(w) is minimized.
        # If label smoothing value is set to zero, the loss
        # is equivalent to NLLLoss or CrossEntropyLoss.
        # All non-true labels are uniformly set to low-confidence.

        target_one_hot = F.one_hot(target, num_classes).float()
        target_one_hot[target_one_hot == 1] = 1 - self.smooth_factor
        target_one_hot[target_one_hot == 0] = self.smooth_factor
        return target_one_hot

    def forward(self, input, target):
        logp = F.log_softmax(input, dim=1)
        target_one_hot = self._smooth_labels(input.size(1), target)
        return F.kl_div(logp, target_one_hot, reduction='sum')


def get_loss(name_loss, t1 = 0.7, t2 = 2):
    LEVELS = {
        'label_smooth_cross_entropy': LabelSmoothingLoss(),
        'tempared_log_loss': TemperedLogLoss(t1 = t1, t2 = t2),
    }

    return LEVELS[name_loss]