from collections import defaultdict
import warnings
from numpy.lib.arraysetops import isin
import torch
import os
import logging
import datetime
import numpy as np
import torch
from tqdm import tqdm
from typing import *
from .utils import F1Record, Record


class Worker(object):

    def __init__(self, opts):
        super().__init__()
        self.train_epoch = opts.train_epoch
        self.no_gpu = opts.no_gpu
        self.gpu = opts.gpu
        self.distributed = self.gpu.count(",") > 0
        self.save_model = opts.save_model
        self.load_model = opts.load_model
        self.log = opts.log
        log_dirs = os.path.split(self.log)[0]
        if not os.path.exists(log_dirs):
            os.makedirs(log_dirs)
        self.log_dir = log_dirs
        logging.basicConfig(filename=self.log, level=logging.INFO)
        self._log = logging.info
        self.epoch = 0
        self.optimization_step = 0
        self.epoch_outputs = dict()
        metric = getattr(opts, 'metric', 'f1')
        if metric == 'f1':
            self.metric = F1Record
        else:
            self.metric = Record
        accumulation_steps = getattr(opts, 'accumulation_steps', 1)
        self.accumulation_step = accumulation_steps
        self.accumulation_pool = []
        self.max_grad_norm = opts.max_grad_norm
        self.train_iterator = None
        self.train_last_it = -1
        self.patience = opts.patience
        self.adv_training = opts.adv

    @classmethod
    def from_options(cls, train_epoch:int, no_gpu:bool, gpu:int, save_model:str, load_model:str, log:str):
        class Opts:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        opts = Opts(
            train_epoch = train_epoch,
            no_gpu = no_gpu,
            gpu = gpu,
            save_model = save_model,
            load_model = load_model,
            log = log)
        return cls(opts)

    @classmethod
    def _to_device(cls, instance:Union[torch.Tensor,List[torch.Tensor],Tuple[torch.Tensor,...],Dict[Any,torch.Tensor]], device:Union[torch.device, None]=None):
        if isinstance(instance, list):
            return [cls._to_device(t, device) for t in instance]
        elif isinstance(instance, dict):
            return {key: cls._to_device(value, device=device) for key, value in instance.items()}
        elif isinstance(instance, tuple):
            vals = [cls._to_device(value, device=device) for value in instance]
            return type(instance)(*vals)
        elif isinstance(instance, int) or isinstance(instance, float) or isinstance(instance, str):
            return instance
        else:
            try:
                return instance.to(device)
            except Exception as e:
                print(f"{type(instance)} not recognized for cuda")
                raise(e)

    def _train_step(self, model, f_loss, batch, optimizer, scheduler=None):
        output = f_loss(batch)
        if isinstance(output, dict):
            loss = output.pop("loss")
        else:
            loss = output
        if len(loss.size()) >= 1:
            loss = loss.mean()
        loss.backward()
        self.accumulated_steps += 1
        if self.accumulated_steps == self.accumulation_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            self.accumulated_steps = 0
            self.optimization_step += 1
            if scheduler:
                scheduler.step()
        if isinstance(output, dict):
            return loss, output
        else:
            return loss

    def run_one_epoch(self, model, loader, split, f_loss=None, f_metrics=None, optimizer=None, scheduler=None, collect_outputs=None, metric=None, max_steps:int=-1, loss_validation:Union[None, Callable]=lambda t:t>0, **kwargs):
        if f_loss is None:
            f_loss = model.forward
        if split == "train":
            model.train()
            if optimizer is None:
                raise ValueError("training requires valid optimizer")
            optimizer.zero_grad()
            if self.epoch == 0 and self.train_iterator is None:
                for _ in loader:
                    pass
        else:
            model.eval()
        epoch_loss = Record()
        epoch_metric = self.metric()
        record_metric = True
        self.accumulated_steps = 0
        if collect_outputs is not None:
            self.epoch_outputs = {key: [] for key in collect_outputs}
        tot = len(loader)
        iterator = None
        if split != "train" or self.train_iterator is None:
            info = ""
            if kwargs is not None:
                info += (" ".join([f"{k}: {v}" for k, v in kwargs.items()]) + '|')
            if split != "train":
                iterator = tqdm(loader, f"{info} Epoch {self.epoch:3d} / {self.train_epoch:3d}: {split[:5]:5s}|", total=tot, ncols=128)
            else:
                iterator = tqdm(loader, f"{info} Epoch {self.epoch+1:3d} / {self.train_epoch:3d}: {split[:5]:5s}|", total=tot, ncols=128)
            iterator = (enumerate(iterator), iterator)
            if split == "train":
                self.epoch += 1
                self.train_iterator = iterator
        else:
            iterator = self.train_iterator

        last_it = self.train_last_it if split == "train" else -1
        it = -1
        while max_steps < 0 or (it - last_it) < max_steps:
            try:
                it, batch = next(iterator[0])
            except StopIteration as e:
                if split == "train":
                    self.train_iterator = None
                break
            if not self.no_gpu and not self.distributed:
                batch = self._to_device(batch, torch.device(f"cuda:0"))
            if split == "train":
                output = self._train_step(model=model, f_loss=f_loss, batch=batch, optimizer=optimizer, scheduler=scheduler)
                if isinstance(output, tuple):
                    loss, output = output
                else:
                    loss = output

            else:
                with torch.no_grad():
                    output = f_loss(batch)
                if isinstance(output, dict):
                    loss = output.pop("loss")
                    if len(loss.size()) > 0:
                        loss = loss.mean()
                    output = {k: v.cpu() for k, v in output.items()}
                else:
                    loss = output
            model_output = output
            for key in self.epoch_outputs:
                if key in output:
                    self.epoch_outputs[key].append(model_output[key])
            if loss_validation(loss.item()):
                epoch_loss += loss.item()
                if metric in model_output and record_metric:
                    metric_val = model_output[metric]
                    if isinstance(metric_val, torch.Tensor):
                        metric_val = metric_val.numpy()
                        if metric == "f1" and len(metric_val.shape) > 1:
                            metric_val = np.sum(metric_val, axis=0)
                        if metric == "accuracy" and len(metric_val.shape) > 0:
                            metric_val = np.sum(metric_val, axis=0)
                    epoch_metric += metric_val
                else:
                    # epoch_metric += f_metrics(model_output)
                    record_metric = False
            else:
                print(f"something goes wrong. {loss.item()}")
            if record_metric:
                postfix = {"loss": f"{epoch_loss}", "metric": f"{epoch_metric}"}
            else:
                postfix = {"loss": f"{epoch_loss}"}
            iterator[1].set_postfix(postfix)
            if max_steps > 0 and it - last_it == max_steps:
                break
        if split == "train":
            if self.train_iterator is None:
                self.train_last_it = -1
            else:
                self.train_last_it = it

        return epoch_loss, epoch_metric if record_metric else None

    def save(self,
        model:Union[torch.nn.Module, Dict],
        optimizer:Union[torch.optim.Optimizer, Dict, None]=None,
        scheduler:Union[torch.optim.lr_scheduler._LRScheduler, Dict, None]=None,
        postfix:str=""):

        save_dirs = self.log_dir
        if not os.path.exists(save_dirs):
            os.makedirs(save_dirs)
        def get_state_dict(x):
            if x is None:
                return None
            elif isinstance(x, dict):
                return x
            else:
                try:
                    return x.state_dict()
                except Exception as e:
                    raise ValueError(f"model, optimizer or scheduler to save must be either a dict or have callable state_dict method")
        if postfix != "":
            save_model = os.path.join(save_dirs, f"{self.save_model}.{postfix}")
        else:
            save_model = os.path.join(save_dirs, self.save_model)
        torch.save({
            "state_dict": get_state_dict(model),
            "optimizer_state_dict": get_state_dict(optimizer),
            "scheduler_state_dict": get_state_dict(scheduler),
            "iter": self.epoch + 1
            },
            save_model
        )

    def load(self, model:torch.nn.Module, optimizer:Union[torch.optim.Optimizer, None]=None, scheduler:Union[torch.optim.lr_scheduler._LRScheduler,None]=None, path:Union[str, None]=None, load_iter:bool=True, strict:bool=True) -> None:
        if path is None:
            path = self.load_model
        if not os.path.exists(path):
            raise FileNotFoundError(f"the path {path} to saved model is not correct")

        state_dict = torch.load(path, map_location=torch.device('cuda:0') if torch.cuda.is_available() and (not self.no_gpu) else torch.device('cpu'))
        print(strict)
        model.load_state_dict(state_dict=state_dict["state_dict"], strict=strict)
        if load_iter:
            self.epoch = state_dict["iter"] - 1
        if optimizer:
            optimizer.load_state_dict(state_dict=state_dict["optimizer_state_dict"])
        if scheduler:
            scheduler.load_state_dict(state_dict=state_dict["scheduler_state_dict"])
        return None

class AdvWorker(Worker):

    def __init__(self, opts):
        super().__init__(opts)
        self.stored_features_t = []
        self.stored_features_s = []
        self.d_steps_per_g_step = getattr(opts, "d_steps_per_g_step", 5)
        self.next_optimization = 'g'
        self.alter = 0

    def _apply(self, model, func:str, *args, **kwargs):
        if hasattr(model, func):
            return getattr(model, func)(*args, **kwargs)
        else:
            return getattr(self, func)(model, *args, **kwargs)

    def compute_features(self, model, batch_l, batch_u, add_d_inputs=True):
        # features_t, features_s = model.get_features(batch_l, batch_u)
        features_t, features_s = model.forward(batch_l=batch_l, batch_u=batch_u, mode='feat')
        if add_d_inputs:
            self.stored_features_t.append([ts.detach() for ts in features_t])
            self.stored_features_s.append([ts.detach() for ts in features_s])
        return features_t, features_s

    def clear_d_inputs(self,):
        self.stored_features_t.clear()
        self.stored_features_s.clear()

    def _d_step(self, features_t, features_s, model, optimizer, scheduler=None):
        # note that we don't accumulate steps in d since it is stored
        # model.clip_params()
        # loss = model.d_forward(features_t, features_s)
        loss = model.forward(features_t=features_t, features_s=features_s, mode="d")
        if len(loss.size()) >= 1:
            loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    def _d_episode(self, model, optimizer, scheduler=None):
        stored_features_t = [
            torch.cat([t[0] for t in self.stored_features_t], dim=0).to(torch.device("cuda:0")),
            torch.cat([t[1] for t in self.stored_features_t], dim=0).to(torch.device("cuda:0"))
        ]
        stored_features_s = [
            torch.cat([t[0] for t in self.stored_features_s], dim=0).to(torch.device("cuda:0")),
            torch.cat([t[1] for t in self.stored_features_s], dim=0).to(torch.device("cuda:0"))
        ]
        for i in range(self.d_steps_per_g_step):
            self._d_step(stored_features_t, stored_features_s, model, optimizer, scheduler)
        self.clear_d_inputs()
        self.next_optimization = 'g'

    def _g_step(self, batch_l, batch_u, features_t, features_s, model, optimizer, scheduler=None, g_params=None):
        warmup = not self.adv_training
        self_training = False#self.epoch > 1
        # output = model.g_forward(batch_l, batch_u, features_t, features_s, warmup=warmup, self_training=self_training)
        output = model.forward(batch_l=batch_l, batch_u=batch_u, features_t=features_t, features_s=features_s, warmup=warmup, self_training=self_training, mode='g')
        loss = output.pop("loss")
        if self.accumulation_step > 1:
            loss = loss / self.accumulation_step
        if len(loss.size()) >= 1:
            loss = loss.mean()
        loss.backward(retain_graph=not warmup)
        if "g_loss" in output:
            g_loss = output.pop("g_loss")
            if not warmup:
                if self.accumulation_step > 1:
                    g_loss = g_loss / self.accumulation_step
                if len(g_loss.size()) >= 1:
                    g_loss = g_loss.mean()
                if g_params is None:
                    g_loss.backward()
                    print('wrong')
                else:
                    g_loss.backward(inputs=g_params)

        self.accumulated_steps += 1
        if self.accumulated_steps == self.accumulation_step:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            self.accumulated_steps = 0
            self.optimization_step += 1
            if scheduler:
                scheduler.step()
            self.next_optimization = 'd'
        if isinstance(output, dict):
            return loss, output
        else:
            return loss
    
    def _ie_step(self, batch_l, batch_u, features_t, features_s, model):
        # output = model.g_forward(batch_l, batch_u, features_t, features_s, warmup=warmup, self_training=self_training)
        output = model.forward(batch_l=batch_l, batch_u=batch_u, features_t=features_t, features_s=features_s, warmup=True, self_training=False, mode='g')
        loss = output.pop("loss")
        if self.accumulation_step > 1:
            loss = loss / self.accumulation_step
        if len(loss.size()) >= 1:
            loss = loss.mean()
        loss.backward()
        if isinstance(output, dict):
            return loss, output
        else:
            return loss
    
    def _adv_g_step(self, batch_l, batch_u, model, features_t=None, features_s=None, g_params=None):
        output = model.forward(batch_l=batch_l, batch_u=batch_u, features_t=features_t, features_s=features_s, mode='g_only')
        g_loss = output.pop("g_loss")
        if self.accumulation_step > 1:
            g_loss = g_loss / self.accumulation_step
        if len(g_loss.size()) >= 1:
            g_loss = g_loss.mean()
        if g_params is None:
            g_loss.backward()
            print('wrong')
        else:
            g_loss.backward(inputs=g_params)
    
    def _accumulate_and_optimize(self,optimizer,scheduler):
        self.accumulated_steps += 1
        if self.accumulated_steps == self.accumulation_step:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            self.accumulated_steps = 0
            self.optimization_step += 1
            if scheduler:
                scheduler.step()
            self.next_optimization = 'd'

    def _train_step(self, model, batch, optimizer, scheduler=None, g_params=None):
        batch_l, batch_u = batch
        optimizer_g, optimizer_d = optimizer
        scheduler_g, scheduler_d = scheduler if scheduler is not None else (None, None)
        features_t, features_s = self.compute_features(model, batch_l, batch_u, add_d_inputs=True)
        g_outputs = self._g_step(batch_l, batch_u, features_t, features_s, model, optimizer_g, scheduler_g, g_params)

        if self.next_optimization == 'd':
            self._d_episode(model, optimizer_d, scheduler_d)
        return g_outputs
    
    def _train_step2(self, model, batch, optimizer, scheduler=None, g_params=None):
        batch_l, batch_u = batch
        optimizer_g, optimizer_d = optimizer
        scheduler_g, scheduler_d = scheduler if scheduler is not None else (None, None)
        features_t, features_s = self.compute_features(model, batch_l, batch_u, add_d_inputs=True)
        g_outputs = self._ie_step(batch_l, batch_u, features_t, features_s, model)
        self._adv_g_step(batch_l, batch_u, model=model, g_params=g_params)
        self._accumulate_and_optimize(optimizer_g, scheduler_g)

        if self.next_optimization == 'd':
            self._d_episode(model, optimizer_d, scheduler_d)
        return g_outputs

    def _get_next(self, iterator, storage):
        item = None
        try:
            item = next(iterator)
        except StopIteration as e:
            iterator = iter(storage)
        return item, iterator

    def train(self, model, loader, optimizer, scheduler=None, max_steps:int=-1, g_params=None, eval_loader=None, f_metric=None, eval_args=None):
        model.train()
        for o in optimizer:
            o.zero_grad()
        loader_l, loader_u = loader
        epoch_loss = Record()
        self.accumulated_steps = 0
        it_l = iter(loader_l); it_u = iter(loader_u)
        self.optimization_step = 0
        self.epoch = 0
        eval_result = [0.5, 0]
        progress = tqdm(desc="Epoch 0", total=max_steps, ncols=100)
        no_better = 0
        while self.optimization_step < max_steps:
            batch_l, it_l = self._get_next(it_l, loader_l)
            if batch_l is None:
                self.epoch += 1
                progress.set_description(f"Epoch {self.epoch}")
                if eval_loader is not None and f_metric is not None:
                    metric = self.eval(model, eval_loader, f_metric, eval_args)
                    if metric[0] > eval_result[0]:
                        eval_result[0] = metric[0]
                        eval_result[1] = metric[1]
                        self.save(model, optimizer, scheduler)
                        print("Current Best", eval_result)
                        no_better = 0
                    else:
                        no_better += 1
                        if no_better == self.patience:
                            break
                    model.train()
                continue
            batch_l = self._to_device(batch_l, torch.device("cuda:0"))
            batch_u, it_u = self._get_next(it_u, loader_u)
            if batch_u is None:
                batch_u, it_u = self._get_next(it_u, loader_u)
            batch_u = self._to_device(batch_u, torch.device("cuda:0"))
            loss, _ = self._train_step(model, [batch_l, batch_u], optimizer, scheduler, g_params)
            if self.accumulated_steps == 0:
                progress.update(1)
            if loss is not None:
                epoch_loss += loss.item()
            postfix = {"loss": f"{epoch_loss}"}
            progress.set_postfix(postfix)
        progress.close()

    def eval(self, model, loader, f_metric, eval_args):
        model.eval()
        if not isinstance(loader, list):
            loader = [loader]


        results = []
        with torch.no_grad():
            for idx, split_loader in enumerate(loader):
                epoch_outputs = defaultdict(list)
                epoch_loss = Record()
                progress = tqdm(split_loader, desc=f"Eval {idx}", ncols=100)
                for batch in progress:
                    outputs = model(batch_l=self._to_device(batch, torch.device("cuda:0")), mode="eval")
                    loss = outputs.pop("loss")
                    epoch_loss += loss.item()
                    postfix = {"loss": f"{epoch_loss}"}
                    progress.set_postfix(postfix)
                    for key, val in outputs.items():
                        epoch_outputs[key].append(val)
                metrics, _ = f_metric(epoch_outputs, eval_args[idx])
                for output_log in [print, self._log]:
                    output_log(f"Eval_{idx}: {metrics.full_result}")
                results.append(metrics.full_result[2])
            # print("Saving")
            # torch.save(epoch_outputs, "./tmp_outputs_en_m_.th")
        return results

    def save(self,
        model:Union[torch.nn.Module, Dict],
        optimizer:Union[List[torch.optim.Optimizer], torch.optim.Optimizer, Dict, None]=None,
        scheduler:Union[List[torch.optim.lr_scheduler._LRScheduler], torch.optim.lr_scheduler._LRScheduler, Dict, None]=None,
        postfix:str=""):

        save_dirs = self.log_dir
        if not os.path.exists(save_dirs):
            os.makedirs(save_dirs)
        def get_state_dict(x):
            if x is None:
                return None
            elif isinstance(x, dict):
                return x
            elif isinstance(x, list):
                return [get_state_dict(t) for t in x]
            else:
                try:
                    return x.state_dict()
                except Exception as e:
                    raise ValueError(f"model, optimizer or scheduler to save must be either a dict or have callable state_dict method")
        if postfix != "":
            save_model = os.path.join(save_dirs, f"{self.save_model}.{postfix}")
        else:
            save_model = os.path.join(save_dirs, self.save_model)
        torch.save({
            "state_dict": get_state_dict(model),
            "optimizer_state_dict": get_state_dict(optimizer),
            "scheduler_state_dict": get_state_dict(scheduler)
            },
            save_model
        )

    def load(self, model:torch.nn.Module, optimizer:Union[torch.optim.Optimizer, None]=None, scheduler:Union[torch.optim.lr_scheduler._LRScheduler,None]=None, path:Union[str, None]=None, strict:bool=True) -> None:
        if path is None:
            path = self.load_model
        if not os.path.exists(path):
            raise FileNotFoundError(f"the path {path} to saved model is not correct")

        state_dict = torch.load(path, map_location=torch.device('cuda:0') if torch.cuda.is_available() and (not self.no_gpu) else torch.device('cpu'))
        model.load_state_dict(state_dict=state_dict["state_dict"], strict=strict)
        if optimizer:
            if isinstance(optimizer, list):
                assert len(optimizer) == len(state_dict["optimizer_state_dict"])
                for o, sd in zip(optimizer, state_dict["optimizer_state_dict"]):
                    o.load_state_dict(state_dict=sd)
            else:
                optimizer.load_state_dict(state_dict=state_dict["optimizer_state_dict"])
        if scheduler:
            if isinstance(optimizer, list):
                assert len(scheduler) == len(state_dict["scheduler_state_dict"])
                for s, sd in zip(scheduler, state_dict["scheduler_state_dict"]):
                    s.load_state_dict(state_dict=sd)
            else:
                scheduler.load_state_dict(state_dict=state_dict["scheduler_state_dict"])
        return None


class AdvTriggerWorker(AdvWorker):

    def __init__(self, opts):
        super().__init__(opts)
        self.stored_features_t = []
        self.stored_features_s = []
        self.d_steps_per_g_step = getattr(opts, "d_steps_per_g_step", 20)
        self.next_optimization = 'g'

    def _apply(self, model, func:str, *args, **kwargs):
        if hasattr(model, func):
            return getattr(model, func)(*args, **kwargs)
        else:
            return getattr(self, func)(model, *args, **kwargs)

    def compute_features(self, model, batch_l, batch_u, add_d_inputs=True):
        features_t, features_s, mask_t, mask_s = model.get_features(batch_l, batch_u)
        if add_d_inputs:
            self.stored_features_t.append([ts[ts_mask>0].detach().cpu() for ts, ts_mask in zip(features_t, mask_t)])
            self.stored_features_s.append([ts[ts_mask>0].detach().cpu() for ts, ts_mask in zip(features_s, mask_s)])
        return features_t, features_s, mask_t, mask_s


    def clear_d_inputs(self,):
        for t in self.stored_features_s:
            t.clear()
        for t in self.stored_features_t:
            t.clear()
        self.stored_features_t = []
        self.stored_features_s = []

    def _d_step(self, features_t, features_s, model, optimizer, scheduler=None):
        # note that we don't accumulate steps in d since it is stored
        # model.clip_params()
        loss = model.d_forward(features_t, features_s)
        if len(loss.size()) >= 1:
            loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    def _d_episode(self, model, optimizer, scheduler=None):
        stored_features_t = [
            torch.cat([t[0] for t in self.stored_features_t], dim=0).to(torch.device("cuda:0")),
            torch.cat([t[1] for t in self.stored_features_t], dim=0).to(torch.device("cuda:0"))
        ]
        stored_features_s = [
            torch.cat([t[0] for t in self.stored_features_s], dim=0).to(torch.device("cuda:0")),
            torch.cat([t[1] for t in self.stored_features_s], dim=0).to(torch.device("cuda:0"))
        ]
        for i in range(self.d_steps_per_g_step):
            self._d_step(stored_features_t, stored_features_s, model, optimizer, scheduler)
        self.clear_d_inputs()
        self.next_optimization = 'g'

    def _g_step(self, batch_l, batch_u, features_t, features_s, mask_t, mask_s, model, optimizer, scheduler=None, g_params=None):
        warmup = not self.adv_training
        self_training = False
        output = model.g_forward(batch_l, batch_u, features_t, features_s, mask_t, mask_s, warmup=warmup, self_training=self_training)
        loss = output.pop("loss")
        if self.accumulation_step > 1:
            loss = loss / self.accumulation_step
        if len(loss.size()) >= 1:
            loss = loss.mean()
        loss.backward(retain_graph=not warmup)
        if "g_loss" in output:
            g_loss = output.pop("g_loss")
            if not warmup:
                if self.accumulation_step > 1:
                    g_loss = g_loss / self.accumulation_step
                if len(g_loss.size()) >= 1:
                    g_loss = g_loss.mean()
                if g_params is None:
                    g_loss.backward()
                    print('wrong')
                else:
                    g_loss.backward(inputs=g_params)

        self.accumulated_steps += 1
        if self.accumulated_steps == self.accumulation_step:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            self.accumulated_steps = 0
            self.optimization_step += 1
            if scheduler:
                scheduler.step()
            self.next_optimization = 'd'
        if isinstance(output, dict):
            return loss, output
        else:
            return loss

    def _train_step(self, model, batch, optimizer, scheduler=None, g_params=None):
        batch_l, batch_u = batch
        optimizer_g, optimizer_d = optimizer
        scheduler_g, scheduler_d = scheduler if scheduler is not None else (None, None)
        features_t, features_s, mask_t, mask_s = self.compute_features(model, batch_l, batch_u, add_d_inputs=True)
        g_outputs = self._g_step(batch_l, batch_u, features_t, features_s, mask_t, mask_s, model, optimizer_g, scheduler_g, g_params)
        if self.next_optimization == 'd':
            self._d_episode(model, optimizer_d, scheduler_d)
        return g_outputs

    def _get_next(self, iterator, storage):
        item = None
        try:
            item = next(iterator)
        except StopIteration as e:
            iterator = iter(storage)
        return item, iterator

    def train(self, model, loader, optimizer, scheduler=None, max_steps:int=-1, g_params=None, eval_loader=None, f_metric=None, eval_args=None):
        model.train()
        for o in optimizer:
            o.zero_grad()
        loader_l, loader_u = loader
        # for t in loader_l:
        #     pass
        # for t in loader_u:
        #     pass
        epoch_loss = Record()
        self.accumulated_steps = 0
        it_l = iter(loader_l); it_u = iter(loader_u)
        self.optimization_step = 0
        self.epoch = 0
        eval_result = [0.2, 0]
        progress = tqdm(desc="Epoch 0", total=max_steps, ncols=100)
        no_better = 0
        while self.optimization_step < max_steps:
            batch_l, it_l = self._get_next(it_l, loader_l)
            if batch_l is None:
                self.epoch += 1
                progress.set_description(f"Epoch {self.epoch}")
                if eval_loader is not None and f_metric is not None:
                    metric = self.eval(model, eval_loader, f_metric, eval_args)
                    if metric[0] > eval_result[0]:
                        eval_result[0] = metric[0]
                        eval_result[1] = metric[1]
                        self.save(model, optimizer, scheduler)
                        print("Current Best", eval_result)
                        no_better = 0
                    else:
                        no_better += 1
                        if no_better == self.patience:
                            break
                    model.train()
                continue
            batch_l = self._to_device(batch_l, torch.device("cuda:0"))
            batch_u, it_u = self._get_next(it_u, loader_u)
            if batch_u is None:
                batch_u, it_u = self._get_next(it_u, loader_u)
            batch_u = self._to_device(batch_u, torch.device("cuda:0"))
            loss, _ = self._train_step(model, [batch_l, batch_u], optimizer, scheduler, g_params)
            if self.accumulated_steps == 0:
                progress.update(1)
            epoch_loss += loss.item()
            postfix = {"loss": f"{epoch_loss}"}
            progress.set_postfix(postfix)
        progress.close()

    def eval(self, model, loader, f_metric, eval_args):
        model.eval()
        if not isinstance(loader, list):
            loader = [loader]


        results = []
        with torch.no_grad():
            for idx, split_loader in enumerate(loader):
                epoch_outputs = defaultdict(list)
                epoch_loss = Record()
                progress = tqdm(split_loader, desc=f"Eval {idx}", ncols=100)
                for batch in progress:
                    outputs = model(batch_l=self._to_device(batch, torch.device("cuda:0")), mode="eval")
                    loss = outputs.pop("loss")
                    epoch_loss += loss.item()
                    postfix = {"loss": f"{epoch_loss}"}
                    progress.set_postfix(postfix)
                    for key, val in outputs.items():
                        epoch_outputs[key].append(val)
                metrics, _ = f_metric(epoch_outputs, eval_args[idx])
                for output_log in [print, self._log]:
                    output_log(f"Eval_{idx}: {metrics.full_result}")
                results.append(metrics.full_result[2])
            # print("Saving")
            # torch.save(epoch_outputs, "./tmp_outputs_zh_n.th")
        return results
