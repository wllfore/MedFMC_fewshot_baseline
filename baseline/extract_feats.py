import os
import torch
import numpy as np
from mmpretrain import get_model, FeatureExtractor
import argparse

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

### extract norm feat via model for each image 
def extract_model_fea(model, img_file):
    inferencer = FeatureExtractor(model, device=DEVICE)
    feat = inferencer(img_file, stage='pre_logits')[0].cpu()
    feat = feat / feat.norm()        
    out_feat = feat.detach().numpy()
    return out_feat

def load_annotations(ann_file):    
    data_infos = []
    with open(ann_file) as f:
        samples = [x.strip() for x in f.readlines()]
        for item in samples:
            filename = item[:-2]
            imglabel = int(item[-1:])            
            info = {}
            info['filename'] = filename
            info['gt_label'] = imglabel
            data_infos.append(info)
    
    return data_infos

def load_endo_annotations(ann_file):    
    data_infos = []
    with open(ann_file) as f:
        samples = [x.strip() for x in f.readlines()]
        for item in samples:
            filename = item[:-8]
            imglabel = item[-7:]
            gt_label = np.asarray(list(map(int, imglabel.split(' '))), dtype=np.int8)
            info = {}
            info['filename'] = filename
            info['gt_label'] = gt_label
            
            data_infos.append(info)
            
    return data_infos

def load_chest_annotations(ann_file):    
    data_infos = []
    with open(ann_file) as f:
        samples = [x.strip() for x in f.readlines()]
        for item in samples:
            filename, imglabel = item.split(' ')
            gt_label = np.asarray(list(map(int, imglabel.split(','))), dtype=np.int8)
            info = {}
            info['filename'] = filename
            info['gt_label'] = gt_label
            data_infos.append(info)
            
    return data_infos

def extract_test_feats(img_list, model, data_dir, fea_dim):
    img_num = len(img_list)

    test_feats = np.zeros((img_num, fea_dim))

    count = 0
    for item in img_list:
        count += 1
        print ('\r', 'process: {}/{}'.format(count, img_num), end='', flush=True)
        img_name = item['filename']
        img_file = os.path.join(data_dir, 'images/', img_name)

        ###add png or jpg 20231005
        img_file = img_file + '.jpg'
        if not os.path.exists(img_file):
            img_file = img_file.replace('.jpg', '.png')
        
        img_fea = extract_model_fea(model, img_file)
        test_feats[count-1, :] = img_fea
        # print (img_fea)     

    return test_feats

def parse_args():
    parser = argparse.ArgumentParser(description='extract feats for test set in MedFMC')
    parser.add_argument('--model', default=None, )                          ### swin-base / simmim-swin-base    
    parser.add_argument('--dataset', default=None)                          ### Retino / ColonPath / NeoJaundice / Endo / ChestDR   
    parser.add_argument('--data_dir', default=None)

    args = parser.parse_args()
    
    return args

def run():
    args = parse_args()  
    print(args)

    model_config = None
    fea_dim = 0
    if args.model == 'swin-base':
        model_config = 'swin-base_16xb64_in1k'
        fea_dim = 1024
    elif args.model == 'simmim-swin-base':
        model_config = 'swin-base-w6_simmim-800e-pre_8xb256-coslr-100e_in1k-192px'
        fea_dim = 1024

    model = get_model(model_config, pretrained=True, device=DEVICE)

    test_list_txt = os.path.join(args.data_dir, 'test.txt')

    test_img_infos = []
    if args.dataset == 'ChestDR':
        test_img_infos = load_chest_annotations(test_list_txt)
    elif args.dataset == 'Endo':
        test_img_infos = load_endo_annotations(test_list_txt)
    else:
        test_img_infos = load_annotations(test_list_txt)
    
    test_img_num = len(test_img_infos)

    if model is not None and test_img_num > 0:
        test_feats = extract_test_feats(test_img_infos, model, args.data_dir, fea_dim)
        ### save test feats
        save_dir = os.path.join(args.data_dir, 'test_feats/')
        make_dir(save_dir)
        np.save(os.path.join(save_dir, args.model + '.npy'), test_feats)

if __name__ == '__main__':
    print ('let us start extract_test_feats now ...')
    run()
    print ('\nfinish ...')   
