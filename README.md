# MedFMC_fewshot_baseline
Implementation of two few-shot method (Baseline and Meta Baseline) for MedFMC

## Requirements
* The following setup has been tested on Python 3.9, Ubuntu 20.04.  
* mmpretrain (recommended 1.0.0rc8): please refer to https://github.com/open-mmlab/mmpretrain for installation details.     
* sklearn (recommended 1.2.2): pip install sklearn  
****
## Usage 
* Firstly, run the script of 'run_extract_feats.sh' to extract the features via pretrained models (e.g. swin-base) of all the test images in each dataset. The extracted features would be stored as '.npy' format file in the sub-folder 'test_feats' of the dataset folder. 

