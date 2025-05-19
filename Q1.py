import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, cell_type='LSTM'):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)

        rnn_cell = getattr(nn, cell_type)  # nn.RNN, nn.LSTM, nn.GRU
        self.rnn = rnn_cell(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout if n_layers > 1 else 0, batch_first = True)

    def forward(self, src):
        # src: [batch_size, src_len]
        embedded = self.embedding(src)  # [batch_size, src_len, emb_dim]
        outputs, hidden = self.rnn(embedded)
        return hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, cell_type='LSTM'):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)

        rnn_cell = getattr(nn, cell_type)
        self.rnn = rnn_cell(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout if n_layers > 1 else 0, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input, hidden):
        # input: [batch_size] -> we process one token at a time
        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.embedding(input)  # [batch_size, 1, emb_dim]

        output, hidden = self.rnn(embedded, hidden)  # output: [batch_size, 1, hid_dim]
        prediction = self.fc_out(output.squeeze(1))  # [batch_size, output_dim]

        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, cell_type='LSTM'):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cell_type = cell_type

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.embedding.num_embeddings

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(src.device)

        hidden = self.encoder(src)

        # Matching output of encoder with what the decoder expects
        if self.decoder.rnn.num_layers > self.encoder.rnn.num_layers:
            diff = self.decoder.rnn.num_layers - self.encoder.rnn.num_layers
            if isinstance(hidden, tuple):  # LSTM
                h, c = hidden
                h = torch.cat([h] + [torch.zeros_like(h[0:1]) for _ in range(diff)], dim=0)
                c = torch.cat([c] + [torch.zeros_like(c[0:1]) for _ in range(diff)], dim=0)
                hidden = (h, c)
            else:  # RNN or GRU
                hidden = torch.cat([hidden] + [torch.zeros_like(hidden) for _ in range(diff)], dim=0)

        if self.encoder.rnn.num_layers > self.decoder.rnn.num_layers:
            if isinstance(hidden, tuple):  # LSTM
                h, c = hidden
                h = h[:self.decoder.rnn.num_layers]
                c = c[:self.decoder.rnn.num_layers]
                hidden = (h, c)
            else:  # RNN or GRU
                hidden = hidden[:self.decoder.rnn.num_layers]

        # Initializing decoder input with <sos> token
        input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:, t] = output

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1

        return outputs
    
    def predict(self, src, trg_stoi, max_len=50):
        self.eval()
        device = next(self.parameters()).device
        if src.dim() == 1:
            src = src.unsqueeze(0)  # add batch dim

        src = src.to(device)
        batch_size = src.size(0)

        with torch.no_grad():
            hidden = self.encoder(src)
            # Matching output of encoder with what the decoder expects
            if self.decoder.rnn.num_layers > self.encoder.rnn.num_layers:
                diff = self.decoder.rnn.num_layers - self.encoder.rnn.num_layers
                if isinstance(hidden, tuple):  # LSTM
                    h, c = hidden
                    h = torch.cat([h] + [torch.zeros_like(h[0:1]) for _ in range(diff)], dim=0)
                    c = torch.cat([c] + [torch.zeros_like(c[0:1]) for _ in range(diff)], dim=0)
                    hidden = (h, c)
                else:  # RNN or GRU
                    hidden = torch.cat([hidden] + [torch.zeros_like(hidden) for _ in range(diff)], dim=0)

            if self.encoder.rnn.num_layers > self.decoder.rnn.num_layers:
                if isinstance(hidden, tuple):  # LSTM
                    h, c = hidden
                    h = h[:self.decoder.rnn.num_layers]
                    c = c[:self.decoder.rnn.num_layers]
                    hidden = (h, c)
                else:  # RNN or GRU
                    hidden = hidden[:self.decoder.rnn.num_layers]

            input_tok = torch.tensor([trg_stoi['<sos>']], device=device)  # start token
            outputs = []

            for _ in range(max_len):
                output, hidden = self.decoder(input_tok.unsqueeze(0), hidden)
                # output shape: [batch, vocab_size]
                output = output.squeeze(0)
                top1 = output.argmax(dim=0)

                if top1.item() == self.trg_stoi['<eos>']:
                    break

                outputs.append(top1.item())
                input_tok = top1

        return outputs  # list of token ids

    def loss_and_accuracy(self, src_stoi, trg_stoi, data_loader, criterion, device):
        self.eval()
        total_loss = 0
        total_sequences = 0
        correct_sequences = 0

        with torch.no_grad():
            for src_batch, trg_batch in data_loader:
                src_batch = src_batch.to(device)
                trg_batch = trg_batch.to(device)

                output = self(src_batch, trg_batch, teacher_forcing_ratio=0.0)
                # output shape: [batch, trg_len, vocab_size]

                batch_size, trg_len, vocab_size = output.shape

                output = output[:, 1:].contiguous()  # skipping <sos> token in predictions
                trg = trg_batch[:, 1:]

                output_flat = output.view(-1, vocab_size)
                trg_flat = trg.contiguous().view(-1)

                loss = criterion(output_flat, trg_flat)
                total_loss += loss.item()
                total_sequences += batch_size

                # For accuracy: get top predictions
                pred_ids = output.argmax(dim=-1)  # [batch, trg_len-1]

                # Compare whole sequences
                for pred_seq, true_seq in zip(pred_ids, trg):
                    # Remove padding and eos before loss calculation
                    special_tokens = {trg_stoi['<pad>'], trg_stoi['<sos>'], trg_stoi['<eos>']}
                    pred_seq_trimmed = [t.item() for t in pred_seq if t.item() not in special_tokens]
                    true_seq_trimmed = [t.item() for t in true_seq if t.item() not in special_tokens]
                    if pred_seq_trimmed == true_seq_trimmed:
                        correct_sequences += 1

        avg_loss_per_seq = total_loss / total_sequences if total_sequences > 0 else 0
        seq_accuracy = correct_sequences / total_sequences if total_sequences > 0 else 0
        return avg_loss_per_seq, seq_accuracy
    
