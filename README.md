# MedFMC_fewshot_baseline
Implementation of two few-shot method (Baseline and Meta Baseline) for MedFMC
****
## Requirements
* The following setup has been tested on Python 3.9, Ubuntu 20.04.  
* mmpretrain (recommended 1.0.0rc8): please refer to https://github.com/open-mmlab/mmpretrain for installation details.     
* sklearn (recommended 1.2.2): pip install sklearn  
****
## Usage 
* Run the script of '**run_extract_feats.sh**' to extract the features via pretrained models (e.g. swin-base) of all the test images in each dataset (ChestDR, Endo, NeoJaudice, ColonPath, Retino). The extracted features would be stored as '.npy' format file in the sub-folder 'test_feats/' of the dataset path (e.g. ./data/ChestDR/test_feats/swin-base.npy).   
* Run the script of '**run_fewshot_baseline.sh**' to test the results of Baseline and Meta Baseline method using 1, 5, 10 shot samples per class under 10 iterations for each dataset. 
* The sampled support-set lists of each dataset corresponding to the reported results in the paper could be found in the package file of each dataset (e.g. ./data/ChestDR/support_set_list.zip).
****
## Dataset Split
* In our experiments, image list used for randomly picking support set of each dataset is saved as 'fewshot-pool.txt', meanwhile image list consisted of the remaining testing images as 'test.txt'. These files can also be found in data folder (e.g. ./data/ChestDR/test.txt).
* When you start to test this baseline, please place all the preprocessed images of each dataset into the sub-folder 'images/' beforehand (e.g. ./data/ChestDR/images/).   
