import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial
import pandas as pd


from prepare import TransliterationDataset,collate_fn,build_vocab
from Q5 import Encoder,Decoder,Seq2Seq,Attention


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Defining the model with the best parameters
INPUT_DIM = len(src_stoi)
OUTPUT_DIM = len(trg_stoi)
EMB_DIM = 256
HID_DIM = 256
N_ENC_LAYERS = 3
N_DEC_LAYERS = 2
CELL_TYPE = 'LSTM'
DROPOUT = 0.3
EPOCHS = 10
LEARNING_RATE = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

attn = Attention(HID_DIM,HID_DIM)
encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, N_ENC_LAYERS, DROPOUT, cell_type=CELL_TYPE)
decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, HID_DIM, N_DEC_LAYERS, DROPOUT, attn, cell_type=CELL_TYPE)
model = Seq2Seq(encoder, decoder, cell_type=CELL_TYPE).to(DEVICE)

# Defining loss & optimizer
criterion = nn.CrossEntropyLoss(ignore_index=trg_stoi['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
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



## Getting predictions for the test dataset and storing it in a folder

latin_texts_test = []
labels_test = []
preds_test = []

with torch.no_grad():
    for src_batch, trg_batch in test_loader:
        src_batch = src_batch.to(device)
        trg_batch = trg_batch.to(device)

        output = model(src_batch, trg_batch, teacher_forcing_ratio=0.0)
        # output shape: [batch, trg_len, vocab_size]

        batch_size, trg_len, vocab_size = output.shape

        output = output[:, 1:].contiguous()  # skip <sos> token in predictions
        src = src_batch[:, :]
        trg = trg_batch[:, 1:]

        output_flat = output.view(-1, vocab_size)
        trg_flat = trg.contiguous().view(-1)

        # For accuracy: get top predictions
        pred_ids = output.argmax(dim=-1)  # [batch, trg_len-1]

        # Compare whole sequences
        for latin_seq, pred_seq, true_seq in zip(src, pred_ids, trg):
            # Remove padding and eos if desired
            special_tokens = {trg_stoi['<pad>'], trg_stoi['<sos>'], trg_stoi['<eos>']}
            latin_seq_trimmed = [t.item() for t in latin_seq if t.item() not in special_tokens]
            pred_seq_trimmed = [t.item() for t in pred_seq if t.item() not in special_tokens]
            true_seq_trimmed = [t.item() for t in true_seq if t.item() not in special_tokens]
            latin_texts_test.append(latin_seq_trimmed)
            labels_test.append(true_seq_trimmed)
            preds_test.append(pred_seq_trimmed)


# Converting list of tokens to strings
for i in range(len(preds_test)):
    for j in range(len(preds_test[i])):
        preds_test[i][j] = trg_itos[preds_test[i][j]]
    preds_test[i] = "".join(preds_test[i])

for i in range(len(labels_test)):
    for j in range(len(labels_test[i])):
        labels_test[i][j] = trg_itos[labels_test[i][j]]
    labels_test[i] = "".join(labels_test[i])

for i in range(len(latin_texts_test)):
    for j in range(len(latin_texts_test[i])):
        latin_texts_test[i][j] = src_itos[latin_texts_test[i][j]]
    latin_texts_test[i] = "".join(latin_texts_test[i])

# Creating a DataFrame with test predictions
df = pd.DataFrame({
    "latin": latin_texts_test,
    "label": labels_test,
    "prediction": preds_test  # remove this column if not needed
})

# Saving as TSV in a predictions_vanilla
df.to_csv("predictions_attention/test_predictions.tsv", sep="\t", index=False)