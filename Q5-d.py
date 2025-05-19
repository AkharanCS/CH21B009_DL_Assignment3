import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
import wandb

from prepare import TransliterationDataset,collate_fn,build_vocab
from Q5 import Encoder,Decoder,Seq2Seq,Attention

# Download the devanagari font using this command before running:
# wget -q https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf -O NotoSansDevanagari-Regular.ttf

# Register the downloaded font
font_path = "./NotoSansDevanagari-Regular.ttf"
devanagari_font = font_manager.FontProperties(fname=font_path)

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



## Plotting the attention heatmap for 9 samples
data_iter = iter(test_loader)
latin, hindi = next(data_iter)

# Taking the first 9 samples
latin = latin[:9]
hindi = hindi[:9]


plt.figure(figsize=(15, 15))
for idx in range(len(latin)):
    src_tensor = latin[idx]
    src_tokens = [src_itos[i.item()] for i in src_tensor if i.item() not in (trg_stoi['<pad>'],)]

    output, attentions = model.predict(src_tensor, trg_stoi)
    trg_tokens = [trg_itos[i] for i in output]

    ax = plt.subplot(3, 3, idx + 1)
    attn_matrix = np.stack(attentions)

    sns.heatmap(attn_matrix, xticklabels=src_tokens, yticklabels=trg_tokens, cmap="viridis", ax=ax)

    # Setting Devanagari font for axis tick labels
    ax.set_xticklabels(src_tokens, rotation=45, ha="right")
    ax.set_yticklabels(trg_tokens, fontproperties=devanagari_font, rotation=0)

    ax.set_xlabel("Input")
    ax.set_ylabel("Output")
    ax.set_title(f"Sample {idx+1}")

plt.tight_layout()
plt.savefig('attention_heatmap.png')
plt.show()

# Logging the heatmap to wandb
wandb.init(project="Assignment3_Q5_heatmap")
wandb.log({"attention_heatmap": wandb.Image('attention_heatmap.png')})