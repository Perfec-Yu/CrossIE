import json

from transformers import BatchEncoding, AutoTokenizer
from models.seq import SeqCls
from models.sent import SentCls
import numpy as np
import os
import torch
from transformers.optimization import AdamW, get_scheduler
from transformers.trainer_pt_utils import get_parameter_names

from utils.utils import F1MetricTag, F1Record, simpleF1
from utils.options import parse_arguments
from utils.worker import Worker


def create_optimizer_and_scheduler(model:torch.nn.Module, learning_rate:float, weight_decay:float, warmup_step:int, train_step:int, adam_beta1:float=0.9, adam_beta2:float=0.999, adam_epsilon:float=1e-8):
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    optimizer_kwargs = {
        "lr": learning_rate,
        "betas": (adam_beta1, adam_beta2),
        "eps": adam_epsilon,
    }
    optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)
    scheduler = get_scheduler(
                "linear",
                optimizer,
                num_warmup_steps=warmup_step,
                num_training_steps=train_step,
            )
    return optimizer, scheduler


def main():
    opts = parse_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{opts.gpu}"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if opts.seed == -1:
        import time
        opts.seed = time.time()
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    if opts.gpu.count(",") > 0:
        opts.batch_size = opts.batch_size * (opts.gpu.count(",")+1)
        opts.eval_batch_size = opts.eval_batch_size * (opts.gpu.count(",")+1)
    if opts.run_task == 'role':
        from utils.role_data import get_data, get_dev_test_encodings
        IEModel = SentCls
    else:
        from utils.data import get_data, get_dev_test_encodings
        IEModel = SeqCls if opts.run_task == 'trigger' else SentCls  
    loaders, label2id = get_data(opts)
    dev_encoding, test_encoding = get_dev_test_encodings(opts)
    nclass = len(label2id['role']) if opts.run_task == 'role' else len(label2id['event'])
    
    F1Metric = F1MetricTag(-1, 0, label2id['event'], AutoTokenizer.from_pretrained(opts.model_name), save_dir=opts.log_dir, fix_span=False) if opts.run_task == 'trigger' else simpleF1

    model = IEModel(
        nclass=nclass,
        model_name=opts.model_name,
        loss_type='sigmoid' if opts.run_task == 'sent' else 'softmax'
    )

    model.to(torch.device('cuda:0') if torch.cuda.is_available() and (not opts.no_gpu) else torch.device('cpu'))

    if opts.gpu.count(",") > 0:
        model = torch.nn.DataParallel(model)
        print('parallel training')


    if not opts.test_only:
        optimizer, scheduler = create_optimizer_and_scheduler(model, opts.learning_rate, opts.decay, opts.warmup_step, len(loaders[0]) * opts.train_epoch // opts.accumulation_steps)
    else:
        optimizer = scheduler = None

    worker = Worker(opts)
    worker._log(str(opts))
    worker._log(json.dumps(label2id))
    if opts.continue_train:
        worker.load(model, optimizer, scheduler)
    elif opts.test_only:
        worker.load(model, strict=False)
    best_test = None
    best_dev = None
    test_metrics = None
    dev_metrics = None
    metric = "f1"
    collect_outputs = {"prediction", "label"}
    termination = False
    patience = opts.patience
    no_better = 0
    total_epoch = 0

    print("start training")
    while not termination:
        if not opts.test_only:
            f_loss = None
            epoch_loss, epoch_metric = worker.run_one_epoch(
                model=model,
                f_loss=f_loss,
                loader=loaders[0],
                split="train",
                optimizer=optimizer,
                scheduler=scheduler,
                metric=metric,
                max_steps=-200)
            total_epoch += 1

            for output_log in [print, worker._log]:
                output_log(
                    f"Epoch {worker.epoch:3d}  Train Loss {epoch_loss} {epoch_metric}")
            worker.run_one_epoch(
                model=model,
                loader=loaders[1],
                split="dev",
                metric=metric,
                max_steps=-1,
                collect_outputs=collect_outputs)
            dev_metrics, _ = F1Metric(worker.epoch_outputs, dev_encoding)
            dev_log = f'Dev {dev_metrics.full_result}|'
            for output_log in [print, worker._log]:
                output_log(f"Epoch {worker.epoch:3d}: {dev_log}")
        else:
            termination = True
        
        

        if opts.test_only:
            worker.run_one_epoch(
                model=model,
                loader=loaders[-1],
                split="test",
                metric=metric,
                collect_outputs=collect_outputs)
            test_metrics, _ = F1Metric(worker.epoch_outputs, test_encoding)
            for output_log in [print, worker._log]:
                output_log(
                    f"Test {test_metrics.full_result}"
                )
        else:
            if (best_dev is None or dev_metrics > best_dev):
                best_dev = dev_metrics
                worker.save(model, optimizer, scheduler, postfix=f"best")
                worker.run_one_epoch(
                    model=model,
                    loader=loaders[-1],
                    split="test",
                    metric=metric,
                    collect_outputs=collect_outputs)
                test_metrics, _ = F1Metric(worker.epoch_outputs, test_encoding)
                for output_log in [print, worker._log]:
                    output_log(
                        f"Test {test_metrics.full_result}"
                    )
                best_test = test_metrics
                no_better = 0
            else:
                no_better += 1
            print(f"Current: {str(dev_metrics)} | History Best:{str(best_dev)} | Patience: {no_better} : {patience}")

            if no_better >= patience or (worker.epoch > worker.train_epoch):
                dev_log = f'{best_dev.full_result},'
                test_log = f'{best_test.full_result},'
                for output_log in [print, worker._log]:
                    output_log(f"BEST DEV : [{dev_log}]")
                    output_log(f"BEST TEST: [{test_log}]")
                termination = True


if __name__ == "__main__":
    main()