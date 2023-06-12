import os
import random
from extract_feats import make_dir
import numpy as np

### Retino 
def gen_support_set_retino(img_list, N_way, K_shot):
    
    support_set = []
    for cls_idx in range(N_way):
        imgs_per_cls = []
        for item in img_list:
            label = item['gt_label']
            if label == cls_idx:
                imgs_per_cls.append(item['filename'])
                
        # print (cls_idx, len(imgs_per_cls), imgs_per_cls[0])
        
        sample_set = None
        if len(imgs_per_cls) <= K_shot:
            sample_set = imgs_per_cls
        else:
            random.shuffle(imgs_per_cls)
            sample_set = imgs_per_cls[:K_shot]
        
        support_set.append(sample_set)            
        
    return support_set

### NeoJaundice / ColonPath
def gen_support_set_twoclass(img_list, K_shot, dataset_type):

    ### sort study_id
    pos_study_ids = []
    neg_study_ids = []
    for item in img_list:
        img_name = item['filename']
        study_id = None
        if dataset_type == 'NeoJaundice':
            study_id = img_name[:4]
        elif dataset_type == 'ColonPath':
            study_id = img_name[:-9]
        label = item['gt_label']

        # print (img_name, study_id, label)

        if label == 1 and study_id not in pos_study_ids:
            pos_study_ids.append(study_id)
        if label == 0 and study_id not in neg_study_ids:
            neg_study_ids.append(study_id)

    # print (len(pos_study_ids), len(neg_study_ids))

    random.shuffle(pos_study_ids)
    random.shuffle(neg_study_ids)

    pick_pos_study_ids = pos_study_ids[:K_shot]
    pick_neg_study_ids = neg_study_ids[:K_shot]

    support_pos_set = []
    support_neg_set = []
    for item in img_list:
        img_name = item['filename']
        study_id = None
        if dataset_type == 'NeoJaundice':
            study_id = img_name[:4]
        elif dataset_type == 'ColonPath':
            study_id = img_name[:-9]

        label = item['gt_label']

        if study_id in pick_pos_study_ids and label == 1:
            support_pos_set.append(img_name)
        if study_id in pick_neg_study_ids and label == 0:
            support_neg_set.append(img_name)

    print (len(support_pos_set), support_pos_set[0])
    print (len(support_neg_set), support_neg_set[0])

    support_set = []
    support_set.append(support_neg_set)
    support_set.append(support_pos_set)

    return support_set

### Endo
def gen_support_set_endo(img_list, N_way, K_shot):

    support_set = []
    for cls_idx in range(N_way):
        study_ids_per_cls = []
        for item in img_list:
            img_name = item['filename']
            label = item['gt_label']
            study_id = img_name[:18]
            if label[cls_idx] == 1 and study_id not in study_ids_per_cls:
                study_ids_per_cls.append(study_id)

        # print (cls_idx, len(study_ids_per_cls))

        random.shuffle(study_ids_per_cls)
        support_study_ids_per_cls = study_ids_per_cls[:K_shot]

        img_list_per_cls = []
        for item in img_list:
            img_name = item['filename']
            label = item['gt_label']
            study_id = img_name[:18]
            if label[cls_idx] == 1 and study_id in support_study_ids_per_cls:
                img_list_per_cls.append([img_name, label])      ### modify for baseline 20230303

        print (cls_idx, len(img_list_per_cls))
        support_set.append(img_list_per_cls)

    return support_set

### ChestDR
def gen_support_set_chest(img_list, N_way, K_shot):
    
    support_set = []
    for cls_idx in range(N_way):
        imgs_per_cls = []
        for item in img_list:
            label = item['gt_label']
            if label[cls_idx] == 1:
                imgs_per_cls.append([item['filename'], label])
                
        # print (cls_idx, len(imgs_per_cls), imgs_per_cls[0])
        
        sample_set = None
        if len(imgs_per_cls) <= K_shot:
            sample_set = imgs_per_cls
        else:
            random.shuffle(imgs_per_cls)
            sample_set = imgs_per_cls[:K_shot]
        
        support_set.append(sample_set)            
        
    return support_set

def gen_support_set_interface(img_list, N_way, K_shot, dataset_type):

    support_set = []

    if dataset_type == 'Retino':
        support_set = gen_support_set_retino(img_list, N_way, K_shot)
    elif dataset_type == 'NeoJaundice' or dataset_type == 'ColonPath':
        support_set = gen_support_set_twoclass(img_list, K_shot, dataset_type)
    elif dataset_type == 'Endo':
        support_set = gen_support_set_endo(img_list, N_way, K_shot)
    elif dataset_type == 'ChestDR':
        support_set = gen_support_set_chest(img_list, N_way, K_shot)

    return support_set

def save_support_list(support_set, dataset_type, save_file):
    fp = open(save_file, 'w')
    cls_num = len(support_set)
    for cls_idx in range(cls_num):
        imgs_per_cls = support_set[cls_idx]
        for item in imgs_per_cls:
            image_name = None
            if dataset_type in ['Retino', 'ColonPath', 'NeoJaundice']:
                image_name = item
            else:
                image_name = item[0]
            
            fp.write('image name: {}, class id: {}\n'.format(image_name, cls_idx))

    fp.close()

def save_fewshot_data(job_dir, dataset_type, support_sets, test_results):
    print ('\nstart save_fewshot_data ...')
    
    ### 1. save support list
    save_list_dir = os.path.join(job_dir, 'support_list/')
    make_dir(save_list_dir)
    count = 0
    for one_set in support_sets:
        # print (item)
        save_list_file = os.path.join(save_list_dir, 'suppot_list_iter_{}.txt'.format(count))
        save_support_list(one_set, dataset_type, save_list_file)
        count += 1

    ### 2. save test result
    save_result_file = os.path.join(job_dir, 'test_result.txt')
    fp = open(save_result_file, 'w')

    all_acc_set = []
    all_auc_set = []
    all_map_set = []

    for one_test in test_results:
        all_auc_set.append(one_test['mean_auc'])
        if dataset_type in ['Retino', 'ColonPath', 'NeoJaundice']:
            all_acc_set.append(one_test['accuracy'])
        else:
            all_map_set.append(one_test['mean_AP'])

    if len(all_acc_set) > 0:
        print ('overall accuracy: mean = {:.3f}, std = {:.3f}'.format(np.mean(all_acc_set), np.std(all_acc_set)))
        fp.write ('overall accuracy: mean = {:.3f}, std = {:.3f}\n'.format(np.mean(all_acc_set), np.std(all_acc_set)))
    if len(all_auc_set) > 0:
        print ('overall AUC: mean = {:.3f}, std = {:.3f}'.format(np.mean(all_auc_set), np.std(all_auc_set)))
        fp.write ('overall AUC: mean = {:.3f}, std = {:.3f}\n'.format(np.mean(all_auc_set), np.std(all_auc_set)))
    if len(all_map_set) > 0:
        print ('overall mAP: mean = {:.3f}, std = {:.3f}'.format(np.mean(all_map_set), np.std(all_map_set)))
        fp.write ('overall mAP: mean = {:.3f}, std = {:.3f}\n'.format(np.mean(all_map_set), np.std(all_map_set)))

    ### save result of each class
    if dataset_type in ['Retino', 'Endo', 'ChestDR']:
        cls_auc_set = []
        cls_map_set = []

        for one_test in test_results:
            cls_auc_set.append(one_test['class_aucs'])
            if dataset_type != 'Retino':
                cls_map_set.append(one_test['class_aps'])

        cls_auc_arr = np.array(cls_auc_set)
        # print (cls_auc_arr.shape, cls_auc_arr)
        for i in range(cls_auc_arr.shape[1]):
            aucs_per_cls = cls_auc_arr[:, i]
            # print (aucs_per_cls)
            print ('class {} auc: mean = {:.3f}, std = {:.3f}'.format(i, np.mean(aucs_per_cls), np.std(aucs_per_cls)))
            fp.write ('class {} auc: mean = {:.3f}, std = {:.3f}\n'.format(i, np.mean(aucs_per_cls), np.std(aucs_per_cls)))


        if len(cls_map_set) > 0:
            cls_map_arr = np.array(cls_map_set)
            for i in range(cls_map_arr.shape[1]):
                maps_per_cls = cls_map_arr[:, i]
                # print (aucs_per_cls)
                print ('class {} mAP: mean = {:.3f}, std = {:.3f}'.format(i, np.mean(maps_per_cls), np.std(maps_per_cls)))
                fp.write ('class {} mAP: mean = {:.3f}, std = {:.3f}\n'.format(i, np.mean(maps_per_cls), np.std(maps_per_cls)))

        


    fp.close()
