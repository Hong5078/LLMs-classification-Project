"""The two required classifiers, both trained from scratch (no pretrained embeddings)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TextCNN(nn.Module):
    """Kim (2014): embedding -> parallel 1D convolutions -> ReLU -> max-over-time -> dropout -> linear."""

    def __init__(self, vocab_size, n_classes, emb_dim, kernel_sizes, n_filters, dropout, pad_id=0):
        super().__init__()
        self.min_len = max(kernel_sizes)
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.convs = nn.ModuleList(nn.Conv1d(emb_dim, n_filters, k) for k in kernel_sizes)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), n_classes)

    def forward(self, ids, lengths=None):
        if ids.size(1) < self.min_len:
            ids = F.pad(ids, (0, self.min_len - ids.size(1)))
        x = self.embedding(ids).transpose(1, 2)                      # (B, emb, T)
        pooled = [F.relu(conv(x)).max(dim=2).values for conv in self.convs]
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


class BiLSTM(nn.Module):
    """Embedding -> (stacked) bidirectional LSTM over packed sequences -> max+mean pooling -> dropout -> linear."""

    def __init__(self, vocab_size, n_classes, emb_dim, hidden, layers, dropout, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(4 * hidden, n_classes)

    def forward(self, ids, lengths):
        x = self.dropout(self.embedding(ids))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = pad_packed_sequence(self.lstm(packed)[0], batch_first=True, total_length=ids.size(1))
        mask = (torch.arange(ids.size(1), device=ids.device)[None, :] < lengths[:, None]).unsqueeze(-1)
        max_pool = out.masked_fill(~mask, float("-inf")).max(dim=1).values
        mean_pool = (out * mask).sum(dim=1) / lengths[:, None].clamp(min=1)
        return self.fc(self.dropout(torch.cat([max_pool, mean_pool], dim=1)))


def build_model(arch, arch_cfg, vocab_size, n_classes, pad_id=0):
    if arch == "cnn":
        return TextCNN(vocab_size, n_classes, arch_cfg["emb_dim"], arch_cfg["kernel_sizes"],
                       arch_cfg["n_filters"], arch_cfg["dropout"], pad_id)
    if arch == "lstm":
        return BiLSTM(vocab_size, n_classes, arch_cfg["emb_dim"], arch_cfg["hidden"],
                      arch_cfg["layers"], arch_cfg["dropout"], pad_id)
    raise ValueError(arch)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
