from sklearn.metrics import confusion_matrix, f1_score, fbeta_score, cohen_kappa_score
from catalyst.dl import AccuracyCallback, MetricCallback, State, Callback, CriterionCallback, OptimizerCallback
from pytorch_toolbelt.utils.torch_utils import to_numpy
from catalyst.core.callback import Callback, CallbackOrder

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

class CappaScoreCallback(Callback):
    def __init__(self,
                 input_key: str = "targets",
                 output_key: str = "logits",
                 prefix: str = "kappa",
                 optimize_thresholds=True,
                 from_regression=False,
                 ignore_index=-100,
                 class_names=None):
        """
        :param input_key: input key to use for precision calculation; specifies our `y_true`.
        :param output_key: output key to use for precision calculation; specifies our `y_pred`.
        """
        super().__init__(CallbackOrder.metric)
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key
        self.targets = []
        self.predictions = []
        self.ignore_index = ignore_index
        self.from_regression = from_regression
        self.class_names = class_names
        self.optimize_thresholds = optimize_thresholds

    def on_loader_start(self, state):
        self.targets = []
        self.predictions = []

    def on_batch_end(self, state):

        targets = to_numpy(state.input[self.input_key].detach())
        outputs = to_numpy(state.output[self.output_key].detach())

        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            outputs = outputs[mask]
            targets = targets[mask]

        self.targets.extend(targets)
        self.predictions.extend(outputs)

    def on_loader_end(self, state: State):
        predictions = to_numpy(self.predictions)
        predictions = np.argmax(predictions, axis=1)     
        score = cohen_kappa_score(predictions, self.targets, weights='quadratic')
        state.loader_metrics[self.prefix] = score






def CosineLoss(input, target, reduction="mean", alpha=1, gamma=2, xent=.1):


    y = torch.Tensor([1]).cuda()
    target = torch.as_tensor(target).cuda()
    input = torch.as_tensor(input).cuda()
    cosine_loss = F.cosine_embedding_loss(input, F.one_hot(target, num_classes=input.size(-1)), y, reduction=reduction)

    cent_loss = F.cross_entropy(F.normalize(input), target, reduce=False)
    pt = torch.exp(-cent_loss)
    focal_loss = (alpha * (1-pt)**gamma * cent_loss).cpu()

    if reduction == "mean":
        focal_loss = torch.mean(focal_loss)

    return cosine_loss + xent * focal_loss


class CosineLossCallback(Callback):
    def __init__(self,
                 input_key: str = "targets",
                 output_key: str = "logits",
                 prefix: str = "cosine_loss"):
        """
        :param input_key: input key to use for precision calculation; specifies our `y_true`.
        :param output_key: output key to use for precision calculation; specifies our `y_pred`.
        """
        super().__init__(CallbackOrder.metric)
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key
        self.targets = []
        self.predictions = []

    def on_loader_start(self, state):
        self.targets = []
        self.predictions = []

    def on_batch_end(self, state):

        targets = to_numpy(state.input[self.input_key].detach())
        outputs = to_numpy(state.output[self.output_key].detach())

        self.targets.extend(targets)
        self.predictions.extend(outputs)

    def on_loader_end(self, state: State):
        # predictions = to_numpy(self.predictions)
        # predictions = np.argmax(predictions, axis=1)     
        score = CosineLoss(self.predictions, self.targets)
        state.loader_metrics[self.prefix] = float(score)