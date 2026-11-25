import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
from transformers import AddedToken, T5TokenizerFast
import torch

PAD_IDX = 0
SQL_OPERATOR_TOKENS = (" <=", " <>", " <")


def create_t5_tokenizer(extend_sql_operators=True, **kwargs):
    """Keep SQL comparison operators representable without changing query text.

    Native T5 has no ``<`` token. Leading spaces retain the supplied SQL's exact
    formatting, while separate compound operators prevent decoding ``< =``.
    These are ordinary output tokens, so skip_special_tokens does not erase them.
    Saved checkpoint tokenizers are loaded independently without re-extending them.
    """
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small", **kwargs)
    if extend_sql_operators:
        tokenizer.add_tokens([AddedToken(text, normalized=False, special=False)
                              for text in SQL_OPERATOR_TOKENS])
    return tokenizer

class T5Dataset(Dataset):

    def __init__(self, data_folder, split, tokenizer=None, max_source_length=None,
                 max_target_length=None, decoder_start_token_id=PAD_IDX):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        # TODO
        if split not in {"train", "dev", "test"}:
            raise ValueError(f"Unknown split: {split}")
        self.tokenizer = tokenizer if tokenizer is not None else create_t5_tokenizer()
        if self.tokenizer.pad_token_id != PAD_IDX:
            raise ValueError("The provided loss convention requires T5 padding token ID 0.")
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.decoder_start_token_id = decoder_start_token_id
        self.split = split
        self.examples = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        # TODO
        self.sources = load_lines(os.path.join(data_folder, f"{split}.nl"))
        self.targets = None if split == "test" else load_lines(os.path.join(data_folder, f"{split}.sql"))
        if not self.sources or any(not source.strip() for source in self.sources):
            raise ValueError(f"{split}.nl must contain nonempty aligned examples.")
        if self.targets is not None:
            if len(self.sources) != len(self.targets):
                raise ValueError(f"Unaligned {split} examples: {len(self.sources)} inputs and {len(self.targets)} targets.")
            if any(not target.strip() for target in self.targets):
                raise ValueError(f"{split}.sql contains an empty target.")
        encoder_ids = tokenizer(self.sources, add_special_tokens=True, padding=False, truncation=False)["input_ids"]
        target_ids = None if self.targets is None else tokenizer(
            self.targets, add_special_tokens=True, padding=False, truncation=False
        )["input_ids"]
        examples = []
        for index, source_ids in enumerate(encoder_ids):
            if self.max_source_length is not None and len(source_ids) > self.max_source_length:
                raise ValueError(f"{split} source {index} has {len(source_ids)} tokens, exceeding {self.max_source_length}; no truncation performed.")
            example = {
                "encoder_ids": torch.tensor(source_ids, dtype=torch.long),
                "decoder_start_id": self.decoder_start_token_id,
            }
            if target_ids is not None:
                target = target_ids[index]
                if not target or target[-1] != tokenizer.eos_token_id:
                    raise ValueError(f"{split} target {index} is missing EOS.")
                if self.max_target_length is not None and len(target) > self.max_target_length:
                    raise ValueError(f"{split} target {index} has {len(target)} tokens, exceeding {self.max_target_length}; no truncation performed.")
                example["decoder_targets"] = torch.tensor(target, dtype=torch.long)
            examples.append(example)
        return examples
    
    def __len__(self):
        # TODO
        return len(self.examples)

    def __getitem__(self, idx):
        # TODO
        return self.examples[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    encoder_ids, encoder_mask, initial_decoder_inputs = test_collate_fn(batch)
    decoder_targets = pad_sequence([example["decoder_targets"] for example in batch], batch_first=True, padding_value=PAD_IDX)
    decoder_inputs = torch.full_like(decoder_targets, PAD_IDX)
    decoder_inputs[:, :1] = initial_decoder_inputs
    decoder_inputs[:, 1:] = decoder_targets[:, :-1]
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    encoder_ids = pad_sequence([example["encoder_ids"] for example in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = encoder_ids.ne(PAD_IDX).long()
    initial_decoder_inputs = torch.tensor([[example["decoder_start_id"]] for example in batch], dtype=torch.long)
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split, data_folder='data', tokenizer=None, max_source_length=None,
                   max_target_length=None, decoder_start_token_id=PAD_IDX):
    dset = T5Dataset(data_folder, split, tokenizer, max_source_length, max_target_length, decoder_start_token_id)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size, data_folder='data', tokenizer=None,
                 max_source_length=None, max_target_length=None, decoder_start_token_id=PAD_IDX):
    if tokenizer is None:
        tokenizer = create_t5_tokenizer()
    options = dict(data_folder=data_folder, tokenizer=tokenizer, max_source_length=max_source_length,
                   max_target_length=max_target_length, decoder_start_token_id=decoder_start_token_id)
    train_loader = get_dataloader(batch_size, "train", **options)
    dev_loader = get_dataloader(test_batch_size, "dev", **options)
    test_loader = get_dataloader(test_batch_size, "test", **options)
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.rstrip('\r\n') for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    if len(train_x) != len(train_y) or len(dev_x) != len(dev_y):
        raise ValueError("Natural-language and SQL rows are not aligned.")
    return train_x, train_y, dev_x, dev_y, test_x
