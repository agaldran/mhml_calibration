# this file was downloaded from https://github.com/GB-TonyLiang/DCA
import torch
import torch.nn.functional as F

def cross_entropy_with_dca_loss(logits, labels, weights=None, alpha=1., beta=5.):
    ce = F.cross_entropy(logits, labels, weight=weights)

    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)
    mean_conf = confidences.float().mean()
    acc = accuracies.float().sum()/len(accuracies)
    dca = torch.abs(mean_conf-acc)
    loss = alpha*ce+beta*dca

    return loss
