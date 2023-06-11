import os
import numpy as np
from extract_feats import extract_model_fea
from sklearn import metrics
from compute_metric import compute_auc, compute_ap
import torch
import math

def compute_cls_centers(support_set, data_dir, model, fea_dim, is_multiclass=True):

    print ('run compute_cls_centers ...')
    
    cls_num = len(support_set)
    cls_centers = []
    for cls_idx in range(cls_num):
        imgs_per_cls = support_set[cls_idx]
        cls_center = torch.zeros(fea_dim)
        for item in imgs_per_cls:
            image_name = None
            if is_multiclass:
                image_name = item
            else:
                image_name = item[0]
            img_file = os.path.join(data_dir, 'images/', image_name)
            img_fea = extract_model_fea(model, img_file)
            cls_center = cls_center + img_fea
        cls_center = cls_center / len(imgs_per_cls)
        cls_center = cls_center / cls_center.norm()
        
        cls_centers.append(cls_center)
    
    # print (cls_centers)

    return cls_centers

def do_meta_test_multiclass(query_set, cls_centers, data_dir, model_name):
    
    print ('run do_meta_test_multiclass ...')

    ### load test feats
    X_Test = np.load(os.path.join(data_dir, 'test_feats/', model_name + '.npy'))
    print (X_Test.shape)

    sample_num = len(query_set)
    cls_num = len(cls_centers)

    correct_num = 0
    cosine_scores_query_set = []
    count = 0
    for item in query_set:    
        img_label = item['gt_label']
        img_fea = X_Test[count, :]
        img_fea = img_fea.astype(np.float32)
        
        max_cosine = 0
        pred_cls = -1
        cosine_scores_per_img = []
        for cls_idx in range(cls_num):
            cls_center = cls_centers[cls_idx]
            # print (type(cls_center), cls_center.shape)
            img_fea_tensor = torch.from_numpy(img_fea)
            # print (type(img_fea_tensor), img_fea_tensor.shape)
            norm_inner = torch.inner(img_fea_tensor, cls_center)
            cosine_simlarity = norm_inner.item()
            # print (cls_idx, cosine_simlarity)
            cosine_scores_per_img.append(cosine_simlarity)
            if max_cosine < cosine_simlarity:
                 max_cosine = cosine_simlarity
                 pred_cls = cls_idx
        
        if img_label == pred_cls:
            correct_num += 1
            
        cosine_scores_query_set.append(cosine_scores_per_img)
        count += 1

    
    acc = correct_num / len(query_set)
            
    # print ('correct_num = ', correct_num, 'pred_accuracy = ', pred_accuracy)
    # print (len(cosine_scores_query_set), cosine_scores_query_set[0])

    ### cpmpute AUC
    gt_labels = np.zeros((sample_num, cls_num))
    for k in range(sample_num):
        sample = query_set[k]
        label = sample['gt_label']
        one_hot_label = np.array([int(i==label) for i in range(cls_num)])
        gt_labels[k, :] = one_hot_label

    cls_scores = np.zeros((sample_num, cls_num))
    for k in range(sample_num):
        cos_score = cosine_scores_query_set[k]
        norm_scores = [math.exp(v) for v in cos_score]
        norm_scores /= np.sum(norm_scores)
        
        cls_scores[k, :] = np.array(norm_scores)
        
    cls_aucs = compute_auc(cls_scores, gt_labels)    
    # print (np.mean(cls_aucs), cls_aucs)
    mean_auc = np.mean(cls_aucs)
    
    return acc, cls_aucs, mean_auc

def do_meta_test_multilabel(query_set, cls_centers, data_dir, model_name):
    
    print ('run do_meta_test_multilabel ...')

    ### load test feats
    X_Test = np.load(os.path.join(data_dir, 'test_feats/', model_name + '.npy'))
    print (X_Test.shape)

    sample_num = len(query_set)
    cls_num = len(cls_centers)

    correct_num = 0
    cosine_scores_query_set = []
    count = 0
    for item in query_set:    
        img_label = item['gt_label']
        img_fea = X_Test[count, :]
        img_fea = img_fea.astype(np.float32)
        
        max_cosine = 0
        cosine_scores_per_img = []
        for cls_idx in range(cls_num):
            cls_center = cls_centers[cls_idx]
            # print (type(cls_center), cls_center.shape)
            img_fea_tensor = torch.from_numpy(img_fea)
            # print (type(img_fea_tensor), img_fea_tensor.shape)
            norm_inner = torch.inner(img_fea_tensor, cls_center)
            cosine_simlarity = norm_inner.item()
            # print (cls_idx, cosine_simlarity)
            cosine_scores_per_img.append(cosine_simlarity)
            
        cosine_scores_query_set.append(cosine_scores_per_img)
        count += 1

    ### cpmpute AUC
    gt_labels = np.zeros((sample_num, cls_num))
    for k in range(sample_num):
        sample = query_set[k]
        label = sample['gt_label']
        # one_hot_label = np.array([int(i==label) for i in range(cls_num)])
        gt_labels[k, :] = label

    cls_scores = np.zeros((sample_num, cls_num))
    for k in range(sample_num):
        cos_score = cosine_scores_query_set[k]
        norm_scores = [math.exp(v) for v in cos_score]
        norm_scores /= np.sum(norm_scores)
        
        cls_scores[k, :] = np.array(norm_scores)
        
    cls_aucs = compute_auc(cls_scores, gt_labels)    
    mean_auc = np.mean(cls_aucs)

    cls_aps = compute_ap(cls_scores, gt_labels)
    mean_AP = np.mean(cls_aps)
    
    return cls_aucs, mean_auc, cls_aps, mean_AP

def do_metabaseline(support_set, query_set, data_dir, model, model_name, fea_dim, cls_num, dataset_type):
    
    test_result = dict()

    if dataset_type in ['Retino', 'ColonPath', 'NeoJaundice']:
        cls_centers = compute_cls_centers(support_set, data_dir, model, fea_dim, is_multiclass=True)
        acc, cls_aucs, mean_auc = do_meta_test_multiclass(query_set, cls_centers, data_dir, model_name)
        # acc, cls_aucs, mean_auc = do_test_multiclass(query_set, data_dir, model_name, clf)
        test_result['accuracy'] = acc
        test_result['mean_auc'] = mean_auc
        test_result['class_aucs'] = cls_aucs
    else:
        cls_centers = compute_cls_centers(support_set, data_dir, model, fea_dim, is_multiclass=False)
        cls_aucs, mean_auc, cls_aps, mean_AP = do_meta_test_multilabel(query_set, cls_centers, data_dir, model_name)
        test_result['mean_auc'] = mean_auc
        test_result['class_aucs'] = cls_aucs
        test_result['mean_AP'] = mean_AP
        test_result['class_aps'] = cls_aps
    
    return test_result