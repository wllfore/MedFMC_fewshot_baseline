import os
import numpy as np
from extract_feats import extract_model_fea
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn import metrics
from compute_metric import compute_auc, compute_ap


def train_cls_model_multiclass(support_set, data_dir, model, fea_dim):
    sample_num = 0
    for cls_set in support_set:
        sample_num += len(cls_set)

    X_train = np.zeros((sample_num, fea_dim))
    cls_num = len(support_set)

    print ('sample_num = ', sample_num, ', cls_num = ', cls_num)

    sample_idx = 0
    Y_train = []
    for cls_idx in range(cls_num):
        imgs_per_cls = support_set[cls_idx]
        for item in imgs_per_cls:
            img_file = os.path.join(data_dir, 'images/', item)
            img_fea = extract_model_fea(model, img_file)
            X_train[sample_idx, :] = img_fea
            sample_idx += 1
            Y_train.append(cls_idx)

    print (X_train.shape, len(Y_train))

    clf = LogisticRegression(random_state=0).fit(X_train, Y_train)
  
    return clf

def train_cls_model_multilabel(support_set, data_dir, model, fea_dim):
    sample_num = 0
    for cls_set in support_set:
        sample_num += len(cls_set)

    print ('sample_num = ', sample_num)

    X_train = np.zeros((sample_num, fea_dim))
    cls_num = len(support_set)
    sample_idx = 0
    Y_train = []
    for cls_idx in range(cls_num):
        imgs_per_cls = support_set[cls_idx]
        for item in imgs_per_cls:
            img_file = os.path.join(data_dir, 'images/', item[0])
            img_fea = extract_model_fea(model, img_file)
            X_train[sample_idx, :] = img_fea
            sample_idx += 1
            Y_train.append(item[1])

    clf = MultiOutputClassifier(LogisticRegression()).fit(X_train, Y_train)

    return clf

def do_test_multiclass(query_set, data_dir, model_name, clf):
    print('run do_test_multiclass ...')

    sample_num = len(query_set)
    print ('sample_num = ', sample_num)

    ### load test feats
    X_Test = np.load(os.path.join(data_dir, 'test_feats/', model_name + '.npy'))
    print (X_Test.shape)
    
    Y_test = []
    sample_idx = 0
    for item in query_set:
        img_label = item['gt_label']
        Y_test.append(img_label)

    prob_Test = clf.predict_proba(X_Test)
    print (prob_Test.shape, prob_Test[0])

    pred_labels = []
    for i in range(sample_num):
        sample_prob = prob_Test[i]
        label = np.argmax(sample_prob)
        pred_labels.append(label)

    cls_num = len(prob_Test[0])
    gt_labels = np.zeros((sample_num, cls_num))
    for k in range(len(query_set)):
        sample = query_set[k]
        label = sample['gt_label']
        one_hot_label = np.array([int(i==label) for i in range(cls_num)])
        gt_labels[k, :] = one_hot_label
    
    ### calculate accuracy and auc for multi-class task
    acc = metrics.accuracy_score(Y_test, pred_labels)

    cls_aucs = compute_auc(prob_Test, gt_labels)
    mean_auc = np.mean(cls_aucs)

    # print ('acc = ', acc, ', auc = ', mean_auc)
    
    return acc, cls_aucs, mean_auc

def do_test_multilabel(query_set, data_dir, model_name, cls_num, clf):
    print('run do_test_multilabel ...')

    sample_num = len(query_set)
    print ('sample_num = ', sample_num)

    ### load test feats
    X_Test = np.load(os.path.join(data_dir, 'test_feats/', model_name + '.npy'))
    print (X_Test.shape)

    gt_labels = np.zeros((sample_num, cls_num))
    sample_idx = 0
    for item in query_set:
        img_label = item['gt_label']
        gt_labels[sample_idx, :] = img_label
        sample_idx += 1

    prob_Test = clf.predict_proba(X_Test)
    # print(len(prob_Test))

    pred_scores = np.zeros((sample_num, cls_num))
    for i in range(cls_num):
        scores_per_cls = prob_Test[i]
        pred_scores[:, i] = scores_per_cls[:, 1]

    ### calculate auc and mAP for multi-label task

    cls_aucs = compute_auc(pred_scores, gt_labels)
    mean_auc = np.mean(cls_aucs)
    
    cls_aps = compute_ap(pred_scores, gt_labels)
    mean_AP = np.mean(cls_aps)

    # print ('mean_auc = ', mean_auc, ', mean_AP = ', mean_AP)

    return cls_aucs, mean_auc, cls_aps, mean_AP


def do_baseline(support_set, query_set, data_dir, model, model_name, fea_dim, cls_num, dataset_type):
    
    test_result = dict()

    if dataset_type in ['Retino', 'ColonPath', 'NeoJaundice']:
        clf = train_cls_model_multiclass(support_set, data_dir, model, fea_dim)
        acc, cls_aucs, mean_auc = do_test_multiclass(query_set, data_dir, model_name, clf)
        test_result['accuracy'] = acc
        test_result['mean_auc'] = mean_auc
        test_result['class_aucs'] = cls_aucs
    else:
        clf = train_cls_model_multilabel(support_set, data_dir, model, fea_dim)
        cls_aucs, mean_auc, cls_aps, mean_AP = do_test_multilabel(query_set, data_dir, model_name, cls_num, clf)
        test_result['mean_auc'] = mean_auc
        test_result['class_aucs'] = cls_aucs
        test_result['mean_AP'] = mean_AP
        test_result['class_aps'] = cls_aps
    
    return test_result
