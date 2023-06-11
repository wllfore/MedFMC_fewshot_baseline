import os
import torch
import numpy as np
import argparse

from mmpretrain import get_model
from extract_feats import extract_model_fea, load_annotations, load_endo_annotations, load_chest_annotations, make_dir
from gen_support_set import gen_support_set_interface, save_fewshot_data
from baseline_func import do_baseline
from metabaseline_func import do_metabaseline

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MedFMC_CLS_NUM = {'ColonPath': 2, 'NeoJaundice': 2, 'Retino': 5, 'Endo': 4, 'ChestDR': 19}


def parse_args():
    parser = argparse.ArgumentParser(description='run few shot baseline on MedFMC')
    parser.add_argument('--method', default=None)                       ### Baseline / MetaBaseline
    parser.add_argument('--model', default=None)                        ### swin-base / simmim-swin-base    
    parser.add_argument('--dataset', default=None)                      ### Retino / ColonPath / NeoJaundice / Endo / ChestDR   
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--shot', default=1, type=int)                  ### 1, 5, 10
    parser.add_argument('--max_iters', default=1, type=int)
    parser.add_argument('--job_name', default=None)

    args = parser.parse_args()
    
    return args


def run():
    args = parse_args()  
    print(args)

    ### step 1. load model
    model_config = None
    fea_dim = 0
    if args.model == 'swin-base':
        model_config = 'swin-base_16xb64_in1k'
        fea_dim = 1024
    elif args.model == 'simmim-swin-base':
        model_config = 'swin-base-w6_simmim-800e-pre_8xb256-coslr-100e_in1k-192px'
        fea_dim = 1024
    model = get_model(model_config, pretrained=True, device=DEVICE)

    ### step 2. load img list
    train_list_txt = os.path.join(args.data_dir, 'trainval.txt')
    test_list_txt = os.path.join(args.data_dir, 'test.txt')

    train_img_infos = []
    test_img_infos = []
    if args.dataset == 'ChestDR':
        train_img_infos = load_chest_annotations(train_list_txt)
        test_img_infos = load_chest_annotations(test_list_txt)
    elif args.dataset == 'Endo':
        train_img_infos = load_endo_annotations(train_list_txt)
        test_img_infos = load_endo_annotations(test_list_txt)
    else:
        train_img_infos = load_annotations(train_list_txt)
        test_img_infos = load_annotations(test_list_txt)

    N_way = MedFMC_CLS_NUM[args.dataset]
    K_shot = args.shot

    print ('N_way = ', N_way, ', K_shot = ', K_shot)


    all_support_sets = []
    all_test_results = []

    for iter in range(args.max_iters):
        print ('\n----------------test iter = {}----------------'.format(iter))
        ### step 3. support set randomly sampled from the trainval set
        support_set = gen_support_set_interface(train_img_infos, N_way, K_shot, args.dataset)
        all_support_sets.append(support_set)

        ### step 4. run test result and compute metrics   
        one_test_res = None
        if args.method == 'Baseline':
            one_test_res = do_baseline(support_set, test_img_infos, args.data_dir, model, args.model, fea_dim, N_way, args.dataset)
        else:
            one_test_res = do_metabaseline(support_set, test_img_infos, args.data_dir, model, args.model, fea_dim, N_way, args.dataset)
        all_test_results.append(one_test_res)

    ### step 5. save support list and test result
    job_dir = os.path.join(args.data_dir, 'work_dir/'+args.job_name)
    make_dir(job_dir)
    save_fewshot_data(job_dir, args.dataset, all_support_sets, all_test_results)


if __name__ == '__main__':
    print ('let us start fewshot baseline test now ...')
    run()
    print ('\nfinish ...')   
