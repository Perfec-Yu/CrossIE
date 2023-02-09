# Cross Lingual IE

### Prepare data

- Prepare data files as sample files in the `sample_data_dir`

### Train Baseline Models

- For Direct Baselines
    - `python run_train.py --src-lang {src_lang} --tgt-lang {tgt_lang} --root {data_dir} --batch-size 8 (or other prefered values) --accumulation-step 1 --gpu {gpu_index} --run_task {role or trigger}`

- For Trans Baselines, for simplicity, we implement this by reusing the adversarial training scripts and simply setting the adversarial loss weight to `0`.
    - `python run_train_adv.py --src-lang {src_lang} --tgt-lang {tgt_lang} --root {data_dir} --batch-size 2 --accumulation-step 4 --gpu {gpu_index} --parallel --adv --adv-training --g-lam 0 --run_task {role or trigger}`

- For Supervised Model in the target language
    - `python run_train.py --src-lang {tgt_lang} --tgt-lang {tgt_lang} --root {data_dir} --batch-size 8 (or other prefered values) --accumulation-step 1 --gpu {gpu_index} --run_task {role or trigger}`

### Train Our Models

- For our models,
    - `python run_train_adv.py --src-lang {src_lang} --tgt-lang {tgt_lang} --root {data_dir} --batch-size 2 --accumulation-step 4 --gpu {gpu_index} --parallel --adv --adv-training --g-lam {g_lam, 1e-1~1e-2 for role, and 1e-3~1e-5 for trigger} --run_task {role or trigger}`

- Note: there are some necessary redundant command line args for current scripts to run correctly, such as `--adv` and `--adv-training`, due to some inconsistency during code development. We shall clean up such redundancy in the next version.
