import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from functools import partial

from prepare import TransliterationDataset,collate_fn,build_vocab

# Loading the dataset
BATCH_SIZE = 32
train_dataset = TransliterationDataset('dakshina_dataset_v1.0\hi\lexicons\hi.translit.sampled.train.tsv')
dev_dataset = TransliterationDataset('dakshina_dataset_v1.0\hi\lexicons\hi.translit.sampled.dev.tsv')
test_dataset = TransliterationDataset('dakshina_dataset_v1.0\hi\lexicons\hi.translit.sampled.test.tsv')
src_stoi, src_itos = build_vocab(train_dataset.latin)
trg_stoi, trg_itos = build_vocab(train_dataset.hindi)

collate = partial(collate_fn, src_vocab=src_stoi, trg_vocab=trg_stoi)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

data_iter = iter(test_loader)
images, labels = next(data_iter)

# Taking the first 10 samples
images = images[:10]
labels = labels[:10]

print(images)
print(labels)