### script of run fewshot baseline
python fewshot_baseline.py --method Baseline --model swin-base --dataset NeoJaundice --data_dir ***/MedFMC/NeoJaundice --shot 10 --max_iters 10 --job_name test_shot10_base
python fewshot_baseline.py --method MetaBaseline --model swin-base --dataset NeoJaundice --data_dir ***/MedFMC/NeoJaundice --shot 10 --max_iters 10 --job_name test_meta_shot10_base
