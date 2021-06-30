import argparse
from dataclasses import dataclass
from typing import *
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import os
from transformers import PreTrainedTokenizerFast, AutoTokenizer, BatchEncoding
from transformers.tokenization_utils_base import TokenSpan

Entailment = Tuple[str, str]

@dataclass
class Instance(object):
    tokens : Union[List[str], str]
    annotations : List[Tuple[int, int, str]]
    sentence_id : str
    links : Optional[List[List[List]]] = None

    @classmethod
    def from_oneie(cls, oneie):
        if 'sentence' in oneie:
            tokens = oneie['sentence']
        else:
            tokens = oneie['tokens']
        annotations = []
        for event in oneie['event_mentions']:
            start = event['trigger']['start']
            end = event['trigger']['end']
            label = event['event_type']
            annotations.append((start, end, label))
        sentence_id = oneie["sent_id"]
        return cls(tokens=tokens, annotations=annotations, sentence_id=sentence_id)

def _to_instance(data, sentence_id_prefix:Optional[str]=None):
    if sentence_id_prefix is None:
        sentence_id_prefix = ""
    elif not sentence_id_prefix.endswith("_"):
        sentence_id_prefix += "_"
    if 'annotations' in data[0]:
        if 'sentence_id' in data[0]:
            for t in data:
                t["sentence_id"] = f"{sentence_id_prefix}{t['sentence_id']}"
            return [Instance(**t) for t in data]
        else:
            return [Instance(**t, sentence_id=f"{sentence_id_prefix}{i}") for i, t in enumerate(data)]
    else:
        return [Instance.from_oneie(t) for t in data]

class IDataset(Dataset):
    _DEFAULT_SETTING = "token"
    _SEED = 2147483647
    def __init__(self,
        instances:List[Instance],
        label2id:Dict[str, int],
        tokenizer:PreTrainedTokenizerFast,
        setting:Optional[str]=None,
        max_length:Optional[int]=None,
        seed:Optional[int]=None,
        label_ignore:Optional[Union[List, Set, int]]=None,
        *args,
        **kwargs) -> None:
        super().__init__()
        if isinstance(label_ignore, int):
            label_ignore = {label_ignore}
        elif isinstance(label_ignore, list):
            label_ignore = set(label_ignore)
        self.label_ignore = set() if label_ignore is None else label_ignore
        self.label2id = label2id
        self.label_offset = 1 if "NA" in self.label2id else 0
        self.instances = instances
        self.instance_tokenized = isinstance(instances[0].tokens, list)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._index_map = None
        self._sent_labels = None
        self._times = 8
        if setting is None:
            self.setting = self._DEFAULT_SETTING
        else:
            self.setting = setting
        self.entailment = None
        if seed is None:
            seed = self._SEED
        else:
            seed = seed
        self._generator = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, index: int) -> Union[Instance, Tuple[Instance, Entailment]]:
        return self.instances[index]

    def collate_batch(self, batch:List[Union[Instance, Tuple[Instance, Entailment]]]) -> BatchEncoding:
        text = None
        labels = None
        spans = None
        text = [i.tokens for i in batch]
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
        special_token_mask = encoded.pop("special_tokens_mask")
        annotations = [i.annotations for i in batch]
        if self.setting == "span":
            _n_annotations = max(len(t) for t in annotations)
            spans = torch.zeros(len(batch), _n_annotations, encoded.input_ids.size(1), dtype=torch.float)
            labels = torch.empty(len(batch), _n_annotations, dtype=torch.long).fill_(-100)
        elif self.setting == "token":
            labels = torch.zeros_like(encoded.input_ids, dtype=torch.long)
        for ibatch, anns in enumerate(annotations):
            for iann, ann in enumerate(anns):
                start, end, label = ann[:3]
                label_id = self.label2id[label] if label not in self.label_ignore else 0
                if self.instance_tokenized:
                    tok_start = encoded.word_to_tokens(ibatch, start)
                    tok_end = encoded.word_to_tokens(ibatch, end)
                else:
                    tok_start = encoded.char_to_token(ibatch, start)
                    tok_end = encoded.char_to_token(ibatch, end-1)
                if tok_end is not None:
                    if isinstance(tok_start, TokenSpan):
                        tok_start = tok_start.start
                    if isinstance(tok_end, TokenSpan):
                        tok_end = tok_end.start
                    else:
                        tok_end += 1
                    if self.setting == "span":
                        spans[ibatch, iann, tok_start:tok_end] = 1. / (tok_end - tok_start)
                        labels[ibatch, iann] = label_id
                    elif self.setting == "token":
                        labels[ibatch, tok_start:tok_end] = label_id

        encoded["labels"] = labels
        encoded["spans"] = spans
        return encoded

    def collate_fn(self, batch:List[Union[Instance, Tuple[Instance, Entailment]]]) -> BatchEncoding:
        sentence_ids = [t.sentence_id for t in batch]
        input_batch = self.collate_batch(batch)

        remove_keys = [key for key in input_batch if input_batch[key] is None]
        for key in remove_keys:
            input_batch.pop(key)
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

    if opts is not None:
        root = getattr(opts, "root", None) if root is None else root
        dataset = getattr(opts, "dataset", None) if dataset is None else dataset
        model_name = getattr(opts, "model_name", None) if model_name is None else model_name
        setting = getattr(opts, "setting", None) if setting is None else setting
        max_length = getattr(opts, "max_length", None) if max_length is None else max_length
        test_only = getattr(opts, "test_only", None) if test_only is None else test_only
        seed = getattr(opts, "seed", None) if seed is None else seed
    if setting == "span":
        train_file = os.path.join(root, f"{dataset}.train.span.jsonl")
        dev_file = os.path.join(root, f"{dataset}.dev.span.jsonl")
        test_file = os.path.join(root, f"{dataset}.test.span.jsonl")
    else:
        train_file = os.path.join(root, f"{dataset}.train.char.jsonl")
        if not os.path.exists(train_file):
            train_file = os.path.join(root, f"{dataset}.train.jsonl")
        dev_file = os.path.join(root, f"{dataset}.dev.char.jsonl")
        if not os.path.exists(dev_file):
            dev_file = os.path.join(root, f"{dataset}.dev.jsonl")
        test_file = os.path.join(root, f"{dataset}.test.char.jsonl")
        if not os.path.exists(test_file):
            test_file = os.path.join(root, f"{dataset}.test.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    with open(dev_file, "rt") as f:
        dev = _to_instance([json.loads(line) for line in f], "dev")
    with open(test_file, "rt") as f:
        test = _to_instance([json.loads(line) for line in f], "test")
    dev = [tokenizer(
            text=t.tokens,
            max_length=max_length,
            is_split_into_words=isinstance(t.tokens, list),
            add_special_tokens=True,
            padding=True,
            truncation=True) for t in dev]
    test = [tokenizer(
            text=t.tokens,
            max_length=max_length,
            is_split_into_words=isinstance(t.tokens, list),
            add_special_tokens=True,
            padding=True,
            truncation=True) for t in test]
    return dev, test

def get_dataset(
    opts:Optional[argparse.Namespace]=None,
    root:Optional[str]=None,
    dataset:Optional[str]=None,
    model_name:Optional[str]=None,
    setting:Optional[str]=None,
    max_length:Optional[int]=None,
    test_only:Optional[bool]=None,
    seed:Optional[int]=None,
    *args,
    **kwargs) -> Tuple[Union[IDataset, None], Union[IDataset, None], IDataset]:

    if opts is not None:
        root = getattr(opts, "root", None) if root is None else root
        dataset = getattr(opts, "dataset", None) if dataset is None else dataset
        model_name = getattr(opts, "model_name", None) if model_name is None else model_name
        setting = getattr(opts, "setting", None) if setting is None else setting
        max_length = getattr(opts, "max_length", None) if max_length is None else max_length
        test_only = getattr(opts, "test_only", None) if test_only is None else test_only
        seed = getattr(opts, "seed", None) if seed is None else seed
    label_info_file = os.path.join(root, "label_info.json")
    if setting == "span":
        train_file = os.path.join(root, f"{dataset}.train.span.jsonl")
        dev_file = os.path.join(root, f"{dataset}.dev.span.jsonl")
        test_file = os.path.join(root, f"{dataset}.test.span.jsonl")
    else:
        train_file = os.path.join(root, f"{dataset}.train.char.jsonl")
        if not os.path.exists(train_file):
            train_file = os.path.join(root, f"{dataset}.train.jsonl")
        dev_file = os.path.join(root, f"{dataset}.dev.char.jsonl")
        if not os.path.exists(dev_file):
            dev_file = os.path.join(root, f"{dataset}.dev.jsonl")
        test_file = os.path.join(root, f"{dataset}.test.char.jsonl")
        if not os.path.exists(test_file):
            test_file = os.path.join(root, f"{dataset}.test.jsonl")
    load_train_dev_file = not test_only
    build_train_dev_dataset = not test_only

    label_info = label2id = None
    train = dev = test = None

    train_dataset = None
    dev_dataset = None

    print("loading files...")
    with open(label_info_file, "rt") as f:
        label_info = json.load(f)
        label2id = {label: info["id"] for label, info in label_info.items()}; label2id["NA"] = 0
        
    if load_train_dev_file:
        with open(train_file, "rt") as f:
            train = _to_instance([json.loads(line) for line in f], "train")
        with open(dev_file, "rt") as f:
            dev = _to_instance([json.loads(line) for line in f], "dev")
    
    with open(test_file, "rt") as f:
        test = _to_instance([json.loads(line) for line in f], "test")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("building pytorch datasets...")
    if build_train_dev_dataset:
        train_dataset = IDataset(
            instances=train,
            label2id=label2id,
            tokenizer=tokenizer,
            setting=setting,
            max_length=max_length,
            mask_prob=None,
            seed=seed)
        dev_dataset = IDataset(
            instances=dev,
            label2id=label2id,
            tokenizer=tokenizer,
            setting=setting,
            max_length=max_length)
    test_dataset = IDataset(
        instances=test,
        label2id=label2id,
        tokenizer=tokenizer,
        setting=setting,
        max_length=max_length)
    
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
    if shuffle is None:
        shuffle = True
    with open(os.path.join(root, "label_info.json"), "rt") as f:
        label_info = json.load(f)
        label2id = {label: info["id"] for label, info in label_info.items()}; label2id["NA"] = 0

    datasets = get_dataset(
        root=root,
        dataset=dataset,
        model_name=model_name,
        setting=setting,
        max_length=max_length,
        seed=seed,
        test_only=test_only)

    if test_only:
        loaders = [None] * (len(datasets) - 1)
    else:
        loaders = []
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