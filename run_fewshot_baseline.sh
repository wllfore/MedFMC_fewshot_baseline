### sample script of run fewshot baseline on test set of MedFMC
python ./baseline/fewshot_baseline.py --method Baseline --model swin-base --dataset ChestDR --data_dir ./data/ChestDR --shot 10 --max_iters 10 --job_name test_shot10_base
python ./baseline/fewshot_baseline.py --method MetaBaseline --model swin-base --dataset ChestDR --data_dir ./data/ChestDR --shot 10 --max_iters 10 --job_name test_meta_shot10_base
