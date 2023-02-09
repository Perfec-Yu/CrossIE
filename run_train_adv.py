import json
from typing import List, Union

from transformers import BatchEncoding, AutoTokenizer
from models.adv import SentEDFilterAdv, TriggerEDFilter, RelationEDFilter
import numpy as np
import os
import torch
from transformers.optimization import AdamW, get_scheduler
from transformers.trainer_pt_utils import get_parameter_names

from utils.utils import F1MetricTag, F1Record
from utils.options import parse_arguments
from utils.worker import Worker, AdvWorker, AdvTriggerWorker


def create_optimizer_and_scheduler(model:Union[List[torch.nn.Module], torch.nn.Module], learning_rate:float, weight_decay:float, warmup_step:int, train_step:int, adam_beta1:float=0.9, adam_beta2:float=0.999, adam_epsilon:float=1e-8):
    if isinstance(model, torch.nn.Module):
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
    else:
        decay_parameters = []
        for m in model:
            decay_parameters.extend([name for name in get_parameter_names(m, [torch.nn.LayerNorm]) if "bias" not in name])
        optimizer_grouped_parameters = [
            {
                "params": [p for m in model for n, p in m.named_parameters() if n in decay_parameters],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for m in model for n, p in m.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]


    optimizer_grouped_parameters = [t for t in optimizer_grouped_parameters if len(t["params"]) > 0]

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


def create_optimizer_and_scheduler_g(model:torch.nn.Module, g_finetune_key:str, learning_rate:float, weight_decay:float, warmup_step:int, train_step:int, adam_beta1:float=0.9, adam_beta2:float=0.999, adam_epsilon:float=1e-8):

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
    optimizer_grouped_parameters_g = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters and g_finetune_key in n],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters and g_finetune_key in n],
            "weight_decay": 0.0,
        },
    ]

    optimizer_grouped_parameters = [t for t in optimizer_grouped_parameters if len(t["params"]) > 0]
    optimizer_grouped_parameters_g = [t for t in optimizer_grouped_parameters_g if len(t["params"]) > 0]

    optimizer_kwargs = {
        "lr": learning_rate,
        "betas": (adam_beta1, adam_beta2),
        "eps": adam_epsilon,
    }
    optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)
    optimizer_g = AdamW(optimizer_grouped_parameters_g, **optimizer_kwargs)
    scheduler = get_scheduler(
                "linear",
                optimizer,
                num_warmup_steps=warmup_step,
                num_training_steps=train_step,
            )
    return optimizer, optimizer_g, scheduler

def simpleF1(outputs, *args, **kwargs):
    pred = outputs["prediction"]
    label = outputs["label"]
    pred = torch.cat(pred, dim=0)
    label = torch.cat(label, dim=0)
    valid = (label > 0).long()
    nmatch = torch.sum((pred == label).long() * valid)
    ngold = torch.sum(valid)
    npred = torch.sum((pred > 0).long())
    record = F1Record()
    record += np.array((ngold.item(), npred.item(), nmatch.item()))
    return record, None

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
    print("initializing loaders")
    if opts.run_task == 'role':
        from utils.role_data import get_data, get_dev_test_encodings
        IEModel = SentEDFilterAdv
        WorkerModel = AdvWorker
        F1Metric = simpleF1 
    else:
        from utils.data import get_data, get_dev_test_encodings
        IEModel = TriggerEDFilter
        WorkerModel = AdvTriggerWorker
        F1Metric = F1MetricTag(-100, [0], label2id['event'], AutoTokenizer.from_pretrained(opts.model_name), save_dir=opts.log_dir, fix_span=False) 
         
    loaders, label2id = get_data(opts)
    dev_encoding, test_encoding = get_dev_test_encodings(opts)
    print("initialzing model")
    model = IEModel(
        nclass={k:len(v) for k,v in label2id.items()},
        model_name=opts.model_name,
        distributed=opts.gpu.count(",") > 0,
        filter_m=20,
        filter_k=1,
        _g_lambda=opts.g_lam,
        classifier='softmax',
        task_str='role'
    )


    if not opts.test_only:
        optimizer_g, scheduler_g = create_optimizer_and_scheduler(model.adv_g, opts.learning_rate, opts.decay, opts.warmup_step, opts.train_step)
        optimizer_d, scheduler_d = create_optimizer_and_scheduler([model.adv_c_t, model.adv_c_s], opts.learning_rate, opts.decay, opts.warmup_step, opts.train_step)
        optimizer = [optimizer_g, optimizer_d]
        scheduler = [scheduler_g, scheduler_d]
        g_params = [p for name, p in model.adv_g.named_parameters() if "domain.2" in name]
        print("n_g_params", len(g_params))
    else:
        optimizer = scheduler = None

    if opts.gpu.count(",") > 0:
        model = torch.nn.DataParallel(model)
        print('parallel training')

    model.to(torch.device('cuda:0') if torch.cuda.is_available() and (not opts.no_gpu) else torch.device('cpu'))


    worker = WorkerModel(opts)
    worker._log(str(opts))
    worker._log(json.dumps(label2id))
    if opts.continue_train:
        worker.load(model, optimizer, scheduler)
    elif opts.test_only:
        worker.load(model)

    print("start training")

    if not opts.test_only:
        worker.train(
            model=model,
            loader=loaders[0],
            optimizer=optimizer,
            scheduler=scheduler,
            max_steps=opts.train_step,
            eval_loader=loaders[1:],
            f_metric=F1Metric,
            eval_args=[dev_encoding, test_encoding],
            g_params=g_params)
    else:
        worker.eval(
            model=model,
            loader=loaders[-1:],
            f_metric=F1Metric,
            eval_args=[dev_encoding, test_encoding])

if __name__ == "__main__":
    main()
