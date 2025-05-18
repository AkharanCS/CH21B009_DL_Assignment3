import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import yaml
from functools import partial

from prepare import TransliterationDataset,collate_fn,build_vocab
from Q1 import Encoder,Decoder,Seq2Seq


# Function which runs during the wandb sweep
def train():
    wandb.init()
    
    # Getting all the configurations
    config = wandb.config
    
    # 3. Initialize model
    INPUT_DIM = len(src_stoi)
    OUTPUT_DIM = len(trg_stoi)
    EMB_DIM = config.in_embed
    HID_DIM = config.hidden_size
    N_ENC_LAYERS = config.encoder_layers
    N_DEC_LAYERS = config.decoder_layers
    CELL_TYPE = config.cell_type
    DROPOUT = config.dropout

    # Initializing wandb run
    wandb.run.name = f"Input_embedding{EMB_DIM}_Hidden_dimension_{HID_DIM}_enc_layers_{N_ENC_LAYERS}_dec_layers_{N_DEC_LAYERS}_cell_type_{CELL_TYPE}_dropout_{DROPOUT}"
    wandb.run.save()

    EPOCHS = 10
    LEARNING_RATE = 0.0001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, N_ENC_LAYERS, DROPOUT, cell_type=CELL_TYPE)
    decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, N_DEC_LAYERS, DROPOUT, cell_type=CELL_TYPE)
    model = Seq2Seq(encoder, decoder, cell_type=CELL_TYPE).to(DEVICE)

    # 4. Define loss & optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=trg_stoi['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Training loop
    epo = []
    val_loss = []
    val_accuracy = []
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
        val_l,val_a = model.loss_and_accuracy(src_stoi, trg_stoi, dev_loader, criterion, DEVICE)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Val Loss: {val_l:.4f} - Val Accuracy: {val_a:.4f}")
        
        epo.append(epoch+1)
        val_loss.append(val_l)
        val_accuracy.append(val_a)

    # Logging all the metrics to wandb
    for i in range(len(epo)):
        wandb.log({"epochs": epo[i], "val_loss": val_loss[i], "val_accuracy": val_accuracy[i]})
        
    test_l,test_a = model.loss_and_accuracy(src_stoi, trg_stoi, test_loader, criterion, DEVICE)
    wandb.log({"test_loss": test_l, "test_acc": test_a})

if __name__ == "__main__":

    #Loading the dataset
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

    with open("config.yaml", "r") as file:
        sweep_config = yaml.safe_load(file)

    # Defining the wandb sweep
    sweep_id = wandb.sweep(sweep_config, project="Assignment3_Q2")
    wandb.agent(sweep_id,function=train,count=30)