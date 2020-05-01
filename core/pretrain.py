"""Pre-train encoder and classifier for source dataset."""

import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np

import params
from utils import make_variable, save_model


def train_src(encoder, classifier, data_loader, val_loader):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    encoder.train()
    classifier.train()

    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=params.c_learning_rate,
        betas=(params.beta1, params.beta2))
    criterion = nn.BCELoss()

    loss_arr = []

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs_pre):
        for step, (images, labels) in enumerate(data_loader):
            # make images and labels variable
            images = make_variable(images)
            labels = make_variable(labels.squeeze_())

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            preds = classifier(encoder(images))
            loss = criterion(preds, labels)

            # optimize source classifier
            loss.backward()
            optimizer.step()

            # print step info
            if ((step + 1) % params.log_step_pre == 0):
                print("Epoch[{}/{}] Step[{}/{}]: loss= {}"
                      .format(epoch + 1,
                              params.num_epochs_pre,
                              step + 1,
                              len(data_loader),
                              loss.data))

        # eval model on validation set
        if ((epoch + 1) % params.eval_step_pre == 0):
            print("Epoch[{}/{}]"
                      .format(epoch + 1,
                              params.num_epochs_pre), end=":")
            loss = eval_src(encoder, classifier, val_loader)
            loss_arr.append(loss)
            encoder.train()
            classifier.train()
            #if ((len(loss_arr) > 5) and (loss_arr[-1] > loss_arr[-2] 
                #and loss_arr[-2] > loss_arr[-3] and loss_arr[-3] > loss_arr[-4])):
                #break


        # save model parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            save_model(encoder, "ADDA-source-encoder-{}.pt".format(epoch + 1))
            save_model(
                classifier, "ADDA-source-classifier-{}.pt".format(epoch + 1))

    # # save final model
    save_model(encoder, "ADDA-source-encoder-final.pt")
    save_model(classifier, "ADDA-source-classifier-final.pt")

    return encoder, classifier


def eval_src(encoder, classifier, data_loader):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.BCELoss()

    predictions = []
    pred_scores = []
    ls = []

    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels)
        labels = labels.squeeze_()

        preds = classifier(encoder(images))
        loss += criterion(preds, labels).data
        ls.append(labels.data.cpu().numpy())

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
        print("Only one label was reported.")


    loss /= len(data_loader)
    acc = acc.float()
    acc /= len(data_loader.dataset)

    print("AvgLoss= {} AvgAcc= {:2%} F1= {} AUC= {} UnweightedAUC= {}".format(loss, acc, f1, auc, unauc))
    return loss
