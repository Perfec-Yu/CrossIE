import argparse
from dataclasses import dataclass
from typing import *
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from argparse import Namespace
import json
import os
from transformers import PreTrainedTokenizerFast, AutoTokenizer, BatchEncoding
from transformers.tokenization_utils_base import TokenSpan
from collections import namedtuple


FileConfig = namedtuple('FileConfig', ['SRC_LANG', 
    'TGT_LANG', 
    'TRAIN_FILE',
    'TRANS_TRAIN_FILE',
    'TRAIN_NO_ANN_FILE',
    'TRANS_TRAIN_NO_ANN_FILE',
    'DEV_FILE',
    'TRANS_DEV_FILE',
    'TEST_FILE',
    'TRANS_TEST_FILE',
    'ADD_NO_ANN'])


def get_file_names(opts):
    SRC_LANG = opts.src_lang
    TGT_LANG = opts.tgt_lang
    ADD_NO_ANN = opts.adv_training

    TRAIN_FILE = f"{SRC_LANG}-{TGT_LANG}.ace_role.train.json"
    TRAIN_NO_ANN_FILE = f"{TGT_LANG}-{SRC_LANG}.ace_role.train.json"
    DEV_FILE = f"{TGT_LANG}-{SRC_LANG}.ace_role.dev.json"
    TEST_FILE = f"{TGT_LANG}-{SRC_LANG}.ace_role.test.json"
    
    config = FileConfig(
        SRC_LANG=SRC_LANG,
        TGT_LANG=TGT_LANG,
        TRAIN_FILE=TRAIN_FILE,
        DEV_FILE=DEV_FILE,
        TEST_FILE=TEST_FILE,
        TRAIN_NO_ANN_FILE=TRAIN_NO_ANN_FILE,
        ADD_NO_ANN=ADD_NO_ANN
    )
    return config


@dataclass
class Instance(object):
    text:str
    label:str
    @classmethod
    def from_entry(cls, entry, lang:str):
        text = entry[lang]
        label = entry['role']
        return cls(text=text, label=label)
    
    def truncate_(self, max_len:int, tokenizer:PreTrainedTokenizerFast):
        tokens = tokenizer(self.text, add_special_tokens=True)
        if len(tokens["input_ids"]) < max_len: return True

        off_as = self.text.index('<a>')
        off_ae = self.text.index('</a>') + 3
        off_bs = self.text.index('<b>')
        off_be = self.text.index('</b>') + 3

        if off_as < 0: off_as = len(self.text)
        if off_bs < 0: off_bs = len(self.text)


        min_s = min(off_as, off_bs)
        max_e = max(off_bs, off_be)
        if min_s == len(self.text):
            if max_e == -1:
                self.text = self.text[:tokens.token_to_chars(max_len-2).end]
            else:
                n_tok = tokens.char_to_token(max_e)
                start = tokens.token_to_chars(n_tok-max_len+3)



@dataclass
class ParallelInstance(object):
    text1:str
    text2:str
    label:str
    @classmethod
    def from_entry(cls, entry, first_lang:str, second_lang:str):
        text1 = entry[first_lang]
        text2 = entry[second_lang]
        label = entry['role']
        return cls(text1=text1, text2=text2, label=label)


def _to_instance(data, first_lang:str, second_lang:str, parallel=True):
    if parallel:
        instances = [ParallelInstance.from_entry(t, first_lang, second_lang) for t in data]
    else:
        instances = [Instance.from_entry(t, first_lang) for t in data]
    return instances

class IDataset(Dataset):
    _LABEL_TYPES = ['role']
    _SEED = 2147483647
    def __init__(self,
        instances:List[Instance],
        label2id:Dict[str, Dict[str, int]],
        tokenizer:PreTrainedTokenizerFast,
        max_length:Optional[int]=None,
        use_pseudo_labels:bool=False,
        seed:Optional[int]=None,
        eval_lang:Optional[int]=None,
        label_ignore:Optional[Union[Dict, List, Set, int]]=None,
        *args,
        **kwargs) -> None:
        super().__init__()
        if isinstance(label_ignore, int):
            label_ignore = {
                label_type: {label_ignore} 
                for label_type in self._LABEL_TYPES
            }
        elif isinstance(label_ignore, list) or isinstance(label_ignore, set):
            label_ignore = label_ignore = {
                label_type: set(label_ignore)
                for label_type in self._LABEL_TYPES
            }
        self.label_ignore = label_ignore if label_ignore else {
                label_type: set() 
                for label_type in self._LABEL_TYPES
            }
        self.label2id = label2id
        self.instances = instances
        self.instance_tokenized = isinstance(instances[0].text, list)
        self.tokenizer = tokenizer
        self.use_pseudo_labels = use_pseudo_labels
        self.max_length = max_length
        print("max_length", max_length)
        self.seed = seed if seed else self._SEED
        self.eval_lang=eval_lang
        self._generator = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, index: int) -> Instance:
        return self.instances[index]
    
    def collate_batch(self, batch:List[Instance]) -> BatchEncoding:
        def label_str_to_id(label_str):
            label_id = self.label2id['role'][label_str] if label_str in self.label2id["role"] else 0
            return label_id if label_id not in self.label_ignore['role'] else 0

        text = [it.text for it in batch]
        label = [label_str_to_id(it.label) for it in batch]

        encoded:BatchEncoding = self.tokenizer(
            text=text,
            max_length=self.max_length,
            is_split_into_words=self.instance_tokenized,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_special_tokens_mask=True,
            return_tensors='pt'
        )
        _ = encoded.pop("special_tokens_mask")
        prefix = "pseudo_" if self.use_pseudo_labels else ""
        encoded[prefix+'sentence_labels'] = torch.LongTensor(label)
        return encoded

    def collate_fn(self, batch:List[Instance]) -> BatchEncoding:
        input_batch = self.collate_batch(batch)
        remove_keys = [key for key in input_batch if input_batch[key] is None]
        for key in remove_keys:
            input_batch.pop(key)
        return input_batch

class ParallelDataset(Dataset):
    _LABEL_TYPES = ['role']
    _SEED = 2147483647
    def __init__(self,
        instances:List[ParallelInstance],
        label2id:Dict[str, Dict[str, int]],
        tokenizer:PreTrainedTokenizerFast,
        max_length:Optional[int]=None,
        use_pseudo_labels:bool=False,
        seed:Optional[int]=None,
        eval_lang:Optional[int]=None,
        label_ignore:Optional[Union[Dict, List, Set, int]]=None,
        *args,
        **kwargs) -> None:
        super().__init__()
        if isinstance(label_ignore, int):
            label_ignore = {
                label_type: {label_ignore} 
                for label_type in self._LABEL_TYPES
            }
        elif isinstance(label_ignore, list) or isinstance(label_ignore, set):
            label_ignore = label_ignore = {
                label_type: set(label_ignore)
                for label_type in self._LABEL_TYPES
            }
        self.label_ignore = label_ignore if label_ignore else {
                label_type: set() 
                for label_type in self._LABEL_TYPES
            }
        self.label2id = label2id
        self.instances = instances
        self.instance_tokenized = isinstance(instances[0].text1, list)
        self.tokenizer = tokenizer
        self.use_pseudo_labels = use_pseudo_labels
        self.max_length = max_length
        self.seed = seed if seed else self._SEED
        self.eval_lang=eval_lang
        self._generator = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, index: int) -> ParallelInstance:
        return self.instances[index]
    
    def collate_batch(self, batch:List[ParallelInstance]) -> BatchEncoding:
        def label_str_to_id(label_str):
            if label_str in self.label2id['role']:
                label_id = self.label2id['role'][label_str] 
            else:
                label_id = 0
            return label_id if label_id not in self.label_ignore['role'] else 0

        text1 = [it.text1 for it in batch]
        text2 = [it.text2 for it in batch]
        label = [label_str_to_id(it.label) for it in batch]

        encoded1:BatchEncoding = self.tokenizer(
            text=text1,
            max_length=self.max_length,
            is_split_into_words=self.instance_tokenized,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_special_tokens_mask=True,
            return_tensors='pt'
        )
        encoded2:BatchEncoding = self.tokenizer(
            text=text2,
            max_length=self.max_length,
            is_split_into_words=self.instance_tokenized,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_special_tokens_mask=True,
            return_tensors='pt'
        )
        encoded = {key: [encoded1[key], encoded2[key]] for key in encoded1}
        _ = encoded.pop("special_tokens_mask")
        prefix = "pseudo_" if self.use_pseudo_labels else ""
        encoded[prefix+'sentence_labels'] = [torch.LongTensor(label)] * 2
        return encoded

    def collate_fn(self, batch:List[Instance]) -> BatchEncoding:
        input_batch = self.collate_batch(batch)
        remove_keys = [key for key in input_batch if input_batch[key] is None]
        for key in remove_keys:
            input_batch.pop(key)
        if self.eval_lang is not None:
            input_batch["eval_lang"] = self.eval_lang
        return input_batch


def get_dev_test_encodings(
    opts:Optional[argparse.Namespace]=None,
    root:Optional[str]=None,
    dataset:Optional[str]=None,
    model_name:Optional[str]=None,
    setting:Optional[str]=None,
    max_length:Optional[int]=None,
    test_only:Optional[bool]=None,
    seed:Optional[int]=None,
    *args,
    **kwargs):
    config = get_file_names(opts)
    if opts is not None:
        root = getattr(opts, "root", None) if root is None else root
        dataset = getattr(opts, "dataset", None) if dataset is None else dataset
        model_name = getattr(opts, "model_name", None) if model_name is None else model_name
        setting = getattr(opts, "setting", None) if setting is None else setting
        max_length = getattr(opts, "max_length", None) if max_length is None else max_length
        test_only = getattr(opts, "test_only", None) if test_only is None else test_only
        seed = getattr(opts, "seed", None) if seed is None else seed
    dev_file = os.path.join(root, config.DEV_FILE)
    test_file = os.path.join(root, config.TEST_FILE)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    with open(dev_file, "rt") as f:
        dev = _to_instance(json.load(f), first_lang=config.SRC_LANG, second_lang=config.TGT_LANG, parallel=True)
    with open(test_file, "rt") as f:
        test = _to_instance(json.load(f), first_lang=config.SRC_LANG, second_lang=config.TGT_LANG, parallel=True)
    dev = [tokenizer(
            text=t.text2,
            max_length=max_length,
            is_split_into_words=isinstance(t.text2, list),
            add_special_tokens=True,
            padding=True,
            truncation=True) for t in dev]
    test = [tokenizer(
            text=t.text2,
            max_length=max_length,
            is_split_into_words=isinstance(t.text2, list),
            add_special_tokens=True,
            padding=True,
            truncation=True) for t in test]
    return dev, test

def get_datasets(
    opts:Optional[argparse.Namespace]=None,
    root:Optional[str]=None,
    dataset:Optional[str]=None,
    model_name:Optional[str]=None,
    setting:Optional[str]=None,
    max_length:Optional[int]=None,
    parallel:Optional[bool]=None,
    test_only:Optional[bool]=None,
    seed:Optional[int]=None,
    *args,
    **kwargs) -> Tuple[Union[Dataset, None], Union[Dataset, None], Dataset]:
    config = get_file_names(opts)
    if opts is not None:
        root = getattr(opts, "root", None) if root is None else root
        dataset = getattr(opts, "dataset", None) if dataset is None else dataset
        model_name = getattr(opts, "model_name", None) if model_name is None else model_name
        setting = getattr(opts, "setting", None) if setting is None else setting
        max_length = getattr(opts, "max_length", None) if max_length is None else max_length
        parallel = getattr(opts, "parallel", None) if parallel is None else parallel
        test_only = getattr(opts, "test_only", None) if test_only is None else test_only
        seed = getattr(opts, "seed", None) if seed is None else seed
    label_info_file = os.path.join(root, "label_info.json")
    train_file = os.path.join(root, config.TRAIN_FILE)
    train_no_ann_file = os.path.join(root, config.TRAIN_NO_ANN_FILE)
    dev_file = os.path.join(root, config.DEV_FILE)
    test_file = os.path.join(root, config.TEST_FILE)
    print("train:", train_file)
    print("dev:", dev_file)
    print("test:", test_file)
    load_trans = parallel
    load_train_dev_file = not test_only
    build_train_dev_dataset = not test_only

    label_info = label2id = None
    train = dev = test = None

    train_dataset = None
    train_no_ann_dataset = None
    dev_dataset = None

    print("loading files...")
    with open(label_info_file, "rt") as f:
        label_info = json.load(f)
        label2id = {
            label_type: {
                label: info["id"] 
                for label,info in type_info.items()
            }
            for label_type, type_info in label_info.items()
        }
        for label_type in label2id:
            label2id[label_type]["NA"] = 0
        
    if load_train_dev_file:
        with open(train_file, "rt") as f:
            train = _to_instance(json.load(f), first_lang=config.SRC_LANG, second_lang=config.TGT_LANG, parallel=parallel)
        with open(train_no_ann_file, "rt") as f:
            train_no_ann = _to_instance(json.load(f), first_lang=config.SRC_LANG, second_lang=config.TGT_LANG, parallel=parallel)
        with open(dev_file, "rt") as f:
            dev = _to_instance(json.load(f), first_lang=config.SRC_LANG if parallel else config.TGT_LANG, second_lang=config.TGT_LANG, parallel=parallel)
    with open(test_file, "rt") as f:
        test = _to_instance(json.load(f), first_lang=config.SRC_LANG if parallel else config.TGT_LANG, second_lang=config.TGT_LANG, parallel=parallel)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("building pytorch datasets...")
    if build_train_dev_dataset:
        if parallel:
            train_dataset = ParallelDataset(
                instances=train,
                label2id=label2id,
                tokenizer=tokenizer,
                max_length=max_length,
                mask_prob=None,
                eval_lang=None,
                seed=seed)
            if config.ADD_NO_ANN:
                train_no_ann_dataset = ParallelDataset(
                    instances=train_no_ann,
                    label2id=label2id,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    mask_prob=None,
                    seed=seed)
            dev_dataset = ParallelDataset(
                instances=dev,
                label2id=label2id,
                tokenizer=tokenizer,
                max_length=512,
                eval_lang=1)
        else:
            train_dataset = IDataset(
                instances=train,
                label2id=label2id,
                tokenizer=tokenizer,
                max_length=max_length,
                mask_prob=None,
                seed=seed)
            dev_dataset = IDataset(
                instances=dev,
                label2id=label2id,
                tokenizer=tokenizer,
                max_length=512)
    test_dataset = (ParallelDataset if parallel else IDataset)(
        instances=test,
        label2id=label2id,
        tokenizer=tokenizer,
        max_length=512,
        eval_lang=1)
    
    if train_no_ann_dataset:
        return [train_dataset, train_no_ann_dataset], dev_dataset, test_dataset
    else:
        return train_dataset, dev_dataset, test_dataset

def get_data(
    opts:Optional[argparse.Namespace]=None,
    batch_size:Optional[int]=None,
    eval_batch_size:Optional[int]=None,
    num_workers:Optional[int]=None,
    seed:Optional[int]=None,
    root:Optional[str]=None,
    dataset:Optional[str]=None,
    model_name:Optional[str]=None,
    setting:Optional[str]=None,
    max_length:Optional[int]=None,
    parallel:Optional[bool]=None,
    test_only:Optional[bool]=None,
    shuffle:Optional[bool]=None,
    *args,
    **kwargs):
    _default_num_workers = 0
    _default_seed = 44739242
    if opts is not None:
        root = getattr(opts, "root", None) if root is None else root
        dataset = getattr(opts, "dataset", None) if dataset is None else dataset
        batch_size = getattr(opts, "batch_size", None) if batch_size is None else batch_size
        eval_batch_size = getattr(opts, "eval_batch_size", batch_size) if eval_batch_size is None else eval_batch_size
        num_workers = getattr(opts, "num_workers", _default_num_workers) if num_workers is None else num_workers
        seed = getattr(opts, "seed", _default_seed) if seed is None else seed
        model_name = getattr(opts, "model_name", None) if model_name is None else model_name
        setting = getattr(opts, "setting", None) if setting is None else setting
        max_length = getattr(opts, "max_length", None) if max_length is None else max_length
        test_only = getattr(opts, "test_only", None) if test_only is None else test_only
        parallel = getattr(opts, "parallel", None) if parallel is None else parallel
    if shuffle is None:
        shuffle = True
    with open(os.path.join(root, "label_info.json"), "rt") as f:
        label_info = json.load(f)
        label2id = {
            label_type: {
                label: info["id"] 
                for label,info in type_info.items()
            }
            for label_type, type_info in label_info.items()
        }
        for label_type in label2id:
            label2id[label_type]["NA"] = 0

    datasets = get_datasets(
        root=root,
        dataset=dataset,
        model_name=model_name,
        setting=setting,
        max_length=max_length,
        seed=seed,
        parallel=parallel,
        test_only=test_only)

    if test_only:
        loaders = [None] * (len(datasets) - 1)
    else:
        loaders = []
        if isinstance(datasets[0], list):
            loaders.append([DataLoader(
                dataset=d,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=True,
                collate_fn=d.collate_fn,
                pin_memory=True,
                num_workers=num_workers,
                generator=torch.Generator().manual_seed(seed)) for d in datasets[0]])
        else:
            loaders.append(DataLoader(
                dataset=datasets[0],
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=False,
                collate_fn=datasets[0].collate_fn,
                pin_memory=True,
                num_workers=num_workers,
                generator=torch.Generator().manual_seed(seed)
            ))
        loaders.extend([DataLoader(
            dataset=d,
            batch_size=batch_size if eval_batch_size <= 0 else eval_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=d.collate_fn,
            pin_memory=True,
            num_workers=num_workers) for d in datasets[1:-1]])
    test_loader = DataLoader(
        dataset=datasets[-1],
        batch_size=batch_size if eval_batch_size <= 0 else eval_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=datasets[-1].collate_fn,
        pin_memory=True,
        num_workers=num_workers)
    loaders.append(test_loader)
    return loaders, label2id