import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

def build_vocab(sequences):
    vocab = set(char for seq in sequences for char in seq)
    stoi = {char: idx + 3 for idx, char in enumerate(sorted(vocab))}
    stoi['<pad>'] = 0
    stoi['<sos>'] = 1
    stoi['<eos>'] = 2
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos

class TransliterationDataset(Dataset):
    def __init__(self, tsv_path):
        df = pd.read_csv(tsv_path, sep='\t', header=None)
        self.hindi = df[0].astype(str).tolist()
        self.latin = df[1].astype(str).tolist()

    def __len__(self):
        return len(self.hindi)

    def __getitem__(self, idx):
        return self.latin[idx], self.hindi[idx]  # (src, trg)


def truncate_at_eos(seq, eos_idx):
    if eos_idx in seq:
        idx = seq.index(eos_idx) + 1
        return seq[:idx]
    return seq

def collate_fn(batch, src_vocab, trg_vocab):
    src_seqs, trg_seqs = zip(*batch)
    src_tensor_list = []
    trg_tensor_list = []

    for src, trg in zip(src_seqs, trg_seqs):
        src_ids = [src_vocab[c] for c in src]
        trg_ids = [trg_vocab['<sos>']] + [trg_vocab[c] for c in trg] + [trg_vocab['<eos>']]
        trg_ids = truncate_at_eos(trg_ids, trg_vocab['<eos>'])

        src_tensor_list.append(torch.tensor(src_ids, dtype=torch.long))
        trg_tensor_list.append(torch.tensor(trg_ids, dtype=torch.long))

    src_padded = pad_sequence(src_tensor_list, batch_first=True, padding_value=src_vocab['<pad>'])
    trg_padded = pad_sequence(trg_tensor_list, batch_first=True, padding_value=trg_vocab['<pad>'])

    return src_padded, trg_padded
