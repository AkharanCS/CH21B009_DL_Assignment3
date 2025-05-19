import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial
from matplotlib import font_manager
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


## Plotting the connectivity diagram for 10 test samples
data_iter = iter(test_loader)
latin, hindi = next(data_iter)

# Taking 10 samples
latin = latin[:10]
hindi = hindi[:10]

# Plotting
fig, axes = plt.subplots(5,2,figsize=(10, 5))
axes = axes.flatten()

# Define special tokens to ignore
special_tokens = {'<pad>','<sos>','<eos>'}

for idx in range(len(latin)):
    ax = axes[idx]
    
    # Filtering source tokens and indices
    src_tokens = [src_itos[i.item()] for i in latin[idx]]
    src_indices = [i.item() for i in latin[idx]]
    src_filtered = [(i, tok) for i, tok in zip(src_indices, src_tokens) if tok not in special_tokens]
    src_x = np.linspace(0.1, 0.9, len(src_filtered))
    y_src = 0.1

    # Filtering target tokens and attention outputs
    output, attentions = model.predict(latin[idx], trg_stoi)
    trg_tokens = [trg_itos[i] for i in output]
    trg_filtered = [(i, tok) for i, tok in zip(output, trg_tokens) if tok not in special_tokens]
    trg_x = np.linspace(0.1, 0.9, len(trg_filtered))
    y_trg = 0.9

    # Plotting source tokens
    for i, (token_id, tok) in enumerate(src_filtered):
        ax.text(src_x[i], y_src - 0.05, tok, ha='center', va='top', fontsize=10)
    
    # Plotting target tokens
    for i, (token_id, tok) in enumerate(trg_filtered):
        ax.text(trg_x[i], y_trg + 0.05, tok, ha='center', va='bottom', fontsize=10, fontproperties=devanagari_font)

    # Adjusting attention matrix to filtered tokens
    src_valid_idx = [i for i, tok in enumerate(src_tokens) if tok not in special_tokens]
    trg_valid_idx = [i for i, tok in enumerate(trg_tokens) if tok not in special_tokens]
    attn_matrix = np.stack(attentions)
    attn_matrix = attn_matrix[np.ix_(trg_valid_idx, src_valid_idx)]

    # Drawing attention lines
    for i, trg_pos in enumerate(trg_x):
        for j, src_pos in enumerate(src_x):
            weight = attn_matrix[i][j]
            if weight > 0.05:  # threshold to avoid clutter
                line = ConnectionPatch(
                    (src_pos, y_src), (trg_pos, y_trg),
                    "data", "data",
                    linewidth=2 * weight,
                    alpha=weight,
                    color="blue"
                )
                ax.add_artist(line)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(f"Sample {idx + 1}", fontsize=5)

plt.tight_layout()
plt.savefig('connectivity.png')
plt.show()

# Logging the connectivity diagram to wandb
wandb.init(project="Assignment3_Q6")
wandb.log({"connectivity": wandb.Image('connectivity.png')})

    