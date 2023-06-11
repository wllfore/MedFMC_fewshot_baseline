import os
import numpy as np
import torch
from sklearn import metrics
from mmpretrain.evaluation import AveragePrecision

### 1. compute auc for each class
def compute_auc(cls_scores, cls_labels):
    cls_aucs = []
    for i in range(cls_scores.shape[1]):
        scores_per_class = cls_scores[:, i]
        labels_per_class = cls_labels[:, i]
        auc_per_class = metrics.roc_auc_score(labels_per_class, scores_per_class)
        # print ('class {} auc = {:.2f}'.format(i+1, auc_per_class*100))
        
        cls_aucs.append(auc_per_class)
    
    return cls_aucs   

### 2. compute AP for each class
def compute_ap(cls_scores, cls_labels):
    cls_aps = []
    for i in range(cls_scores.shape[1]):
        scores_per_class = cls_scores[:, i]
        labels_per_class = cls_labels[:, i]

        scores_per_class = scores_per_class.reshape(cls_scores.shape[0], 1)
        labels_per_class = labels_per_class.reshape(cls_scores.shape[0], 1)

        # print (scores_per_class.shape, labels_per_class.shape)

        ap = AveragePrecision.calculate(scores_per_class, labels_per_class)
        ap_per_class = ap.detach().numpy()

        cls_aps.append(ap_per_class/100)
    
    return cls_aps