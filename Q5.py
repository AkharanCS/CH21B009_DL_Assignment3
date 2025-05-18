import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [batch, dec_hid_dim]
        # encoder_outputs: [batch, src_len, enc_hid_dim]
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)

        # Repeat decoder hidden state across all src_len
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch, src_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((decoder_hidden, encoder_outputs), dim=2)))  # [batch, src_len, dec_hid_dim]
        attention = self.v(energy).squeeze(2)  # [batch, src_len]

        return torch.softmax(attention, dim=1)  # [batch, src_len]


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, cell_type='LSTM'):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        rnn_cell = getattr(nn, cell_type)
        self.rnn = rnn_cell(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout if n_layers > 1 else 0, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        # outputs: [batch, src_len, hid_dim]
        # hidden: [n_layers, batch, hid_dim] or tuple for LSTM
        return outputs, hidden
    

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, dropout, attention, cell_type='GRU'):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = getattr(nn, cell_type)(emb_dim + enc_hid_dim, dec_hid_dim, num_layers=n_layers, dropout=dropout if n_layers > 1 else 0, batch_first=True)
        self.fc_out = nn.Linear(emb_dim + dec_hid_dim + enc_hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(1)  # [batch, 1]
        embedded = self.dropout(self.embedding(input))  # [batch, 1, emb_dim]

        # Get attention weights
        if isinstance(hidden, tuple):  # LSTM
            dec_hidden = hidden[0][-1]  # Use last layer's hidden
        else:
            dec_hidden = hidden[-1]

        attn_weights = self.attention(dec_hidden, encoder_outputs)  # [batch, src_len]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [batch, 1, enc_hid_dim]

        rnn_input = torch.cat((embedded, context), dim=2)  # [batch, 1, emb+enc_hid]
        output, hidden = self.rnn(rnn_input, hidden)  # output: [batch, 1, dec_hid_dim]

        output = output.squeeze(1)
        context = context.squeeze(1)
        embedded = embedded.squeeze(1)

        prediction = self.fc_out(torch.cat((output, context, embedded), dim=1))  # [batch, output_dim]

        return prediction, hidden
    

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, cell_type = "LSTM"):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cell_type = cell_type

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(src.device)

        encoder_outputs, hidden = self.encoder(src)
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


        input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input = trg[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs
        
    
    def predict(self, src, trg_stoi, max_len=50):
        """
        Predict output sequence for a single source sequence tensor.
        
        src: Tensor shape [seq_len] or [1, seq_len]
        Returns: list of predicted token indices (excluding <sos>)
        """
        self.eval()
        device = next(self.parameters()).device
        if src.dim() == 1:
            src = src.unsqueeze(0)  # add batch dim

        src = src.to(device)
        batch_size = src.size(0)

        with torch.no_grad():
            hidden = self.encoder(src)

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

                output = output[:, 1:].contiguous()  # skip <sos> token in predictions
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
                    # Remove padding and eos if desired
                    special_tokens = {trg_stoi['<pad>'], trg_stoi['<sos>'], trg_stoi['<eos>']}
                    pred_seq_trimmed = [t.item() for t in pred_seq if t.item() not in special_tokens]
                    true_seq_trimmed = [t.item() for t in true_seq if t.item() not in special_tokens]
                    #print(pred_seq,true_seq)
                    #print(pred_seq_trimmed,true_seq_trimmed)
                    if pred_seq_trimmed == true_seq_trimmed:
                        correct_sequences += 1

        avg_loss_per_seq = total_loss / total_sequences if total_sequences > 0 else 0
        seq_accuracy = correct_sequences / total_sequences if total_sequences > 0 else 0
        return avg_loss_per_seq, seq_accuracy
    

    
