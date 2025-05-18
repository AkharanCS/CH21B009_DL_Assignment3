import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from functools import partial

# Wrap the collate function with your vocabularies

from prepare import TransliterationDataset,collate_fn,build_vocab
from Q1 import Encoder,Decoder,Seq2Seq

# 1. Define training hyperparameters
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load dataset
train_dataset = TransliterationDataset('dakshina_dataset_v1.0\hi\lexicons\hi.translit.sampled.train.tsv')
dev_dataset = TransliterationDataset('dakshina_dataset_v1.0\hi\lexicons\hi.translit.sampled.dev.tsv')
test_dataset = TransliterationDataset('dakshina_dataset_v1.0\hi\lexicons\hi.translit.sampled.test.tsv')
src_stoi, src_itos = build_vocab(train_dataset.latin)
trg_stoi, trg_itos = build_vocab(train_dataset.hindi)

collate = partial(collate_fn, src_vocab=src_stoi, trg_vocab=trg_stoi)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

# 3. Initialize model
INPUT_DIM = len(src_stoi)
OUTPUT_DIM = len(trg_stoi)
EMB_DIM = 64
HID_DIM = 128
N_LAYERS = 2
CELL_TYPE = 'GRU'

encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, cell_type=CELL_TYPE)
decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, cell_type=CELL_TYPE)
model = Seq2Seq(encoder, decoder, cell_type=CELL_TYPE).to(DEVICE)

# 4. Define loss & optimizer
criterion = nn.CrossEntropyLoss(ignore_index=trg_stoi['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 5. Training loop
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for src, trg in train_loader:
        src, trg = src.to(DEVICE), trg.to(DEVICE)

        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio=0.5)
        # output: [batch_size, trg_len, output_dim]

        # Shift output and target for loss computation
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    val_loss,val_accuracy = model.loss_and_accuracy(src_stoi, trg_stoi, dev_loader, criterion, DEVICE)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")