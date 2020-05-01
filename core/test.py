"""Test script to classify target data."""

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np

from utils import make_variable


def eval_tgt(encoder, classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    #print("setting eval state for Dropout and BN layers")
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0.0
    acc = 0.0

    # set loss function
    criterion = nn.BCELoss()
    predictions = []
    pred_scores = []
    ls = []
    #count = 0
    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels)
        labels = labels.squeeze_()
        #print(count)
        #count += 1

        preds = classifier(encoder(images))
        loss += criterion(preds, labels).data
        ls.append(labels.data.cpu().numpy())
        #print(preds.data.max(1))

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()
        predictions.append(pred_cls.data.cpu().numpy())
        pred_scores.append(preds.data.cpu().max(1)[0])

    ls = np.concatenate(ls)
    z = np.concatenate(pred_scores)
    f1 = f1_score(ls, np.concatenate(predictions))
    try:
        auc = roc_auc_score(ls, z, average='weighted')
        unauc = roc_auc_score(ls, z)
    except:
        auc = 0
        print("Only one label was reported.")

    loss /= len(data_loader)
    acc = acc.float()
    acc /= len(data_loader.dataset)

    print("AvgLoss= {} AvgAcc= {:2%} F1= {} AUC= {} UnweightedAUC= {}".format(loss, acc, f1, auc, unauc))

    return loss
