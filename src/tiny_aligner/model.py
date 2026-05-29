"""
TinyAligner: CTC acoustic model for Chinese + English forced alignment.

Architecture (iPhone-deployable, ~3.3M params with default config):
  Input: [B, T, 40] log Mel-filterbank

  1. Conv stack (3 layers, channels 96/192/256) – local feature extraction,
     time-downsampling ×2.
     Each: Conv1d → BatchNorm → ReLU → (optional stride)

  2. Bi-GRU (2 layers, hidden=320, dropout=0.1) – temporal modeling.

  3. Linear projection → phoneme logits [B, T', n_phones]

  Time downsampling: 2× → 10ms × 2 = 20ms per frame (well under 50ms target)
  Core ML size: ~3.3 MB quantized (INT8).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, dilation=1):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                              padding=pad, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        # x: [B, C, T]
        return F.relu(self.bn(self.conv(x)))


class TinyAligner(nn.Module):
    """
    Tiny CTC acoustic model for Chinese phoneme recognition / forced alignment.
    
    Design choices:
    - No attention (CTC is simpler, more robust for alignment)
    - Conv front-end extracts local patterns (fast on CPU/ANE)
    - Single Bi-GRU for sequential context
    - ~500K params → ~2MB INT8 Core ML model
    
    Why GRU over LSTM: fewer parameters, similar accuracy, faster on iPhone ANE.
    Why Bi-directional: better for offline alignment (full audio available).
    """

    def __init__(self, n_phones: int, n_mels: int = 40, hidden: int = 320,
                 conv_channels: tuple[int, int, int] = (96, 192, 256),
                 num_rnn_layers: int = 2, rnn_dropout: float = 0.1):
        super().__init__()
        self.n_phones = n_phones
        self.n_mels = n_mels
        self.hidden = hidden
        self.conv_channels = tuple(conv_channels)
        self.num_rnn_layers = num_rnn_layers

        c1, c2, c3 = conv_channels

        # Conv feature extractor: input [B, T, 40] → [B, T/2, c3]
        self.conv = nn.Sequential(
            ConvBlock(n_mels, c1, kernel=3, stride=1),     # [B, c1, T]
            ConvBlock(c1, c2, kernel=3, stride=1),         # [B, c2, T]
            ConvBlock(c2, c3, kernel=3, stride=2),         # [B, c3, T/2]  ← 2× downsample
        )

        # Bi-GRU: [B, T/2, c3] → [B, T/2, 2*hidden]
        self.rnn = nn.GRU(
            input_size=c3,
            hidden_size=hidden,
            num_layers=num_rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=rnn_dropout if num_rnn_layers > 1 else 0.0,
        )

        # Output projection: 256 → n_phones
        self.fc = nn.Linear(hidden * 2, n_phones)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, mel: torch.Tensor, lengths: torch.Tensor = None):
        """
        Args:
            mel: [B, T, n_mels] log mel-filterbank
            lengths: [B] actual frame lengths (before padding)
        
        Returns:
            log_probs: [T', B, n_phones]  (CTC convention: time-first)
            out_lengths: [B] output frame lengths after downsampling
        """
        # [B, T, C] → [B, C, T] for Conv1d
        x = mel.transpose(1, 2)
        x = self.conv(x)              # [B, 128, T/2]
        x = x.transpose(1, 2)        # [B, T/2, 128]

        # Pack for RNN efficiency (optional but good practice)
        if lengths is not None:
            out_lengths = (lengths // 2).clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                x, out_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            rnn_out, _ = self.rnn(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        else:
            x, _ = self.rnn(x)
            out_lengths = torch.full((mel.shape[0],), x.shape[1], dtype=torch.long)

        logits = self.fc(x)                        # [B, T/2, n_phones]
        log_probs = F.log_softmax(logits, dim=-1)  # [B, T/2, n_phones]
        log_probs = log_probs.transpose(0, 1)      # [T/2, B, n_phones]  ← CTC format

        return log_probs, out_lengths

    def get_emissions(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Inference mode: get per-frame log probabilities.
        
        Args:
            mel: [T, n_mels] or [1, T, n_mels]
        Returns:
            log_probs: [T', n_phones]
        """
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)  # [1, T, C]
        with torch.no_grad():
            log_probs, _ = self.forward(mel)  # [T', 1, C]
        return log_probs.squeeze(1)  # [T', n_phones]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(n_phones: int, n_mels: int = 40, hidden: int = 320,
                conv_channels: tuple[int, int, int] = (96, 192, 256),
                num_rnn_layers: int = 2, rnn_dropout: float = 0.1) -> TinyAligner:
    model = TinyAligner(n_phones=n_phones, n_mels=n_mels, hidden=hidden,
                        conv_channels=conv_channels,
                        num_rnn_layers=num_rnn_layers, rnn_dropout=rnn_dropout)
    print(f"[Model] TinyAligner: {model.count_parameters():,} parameters "
          f"(conv={conv_channels}, hidden={hidden}, rnn_layers={num_rnn_layers})")
    return model


if __name__ == "__main__":
    # Quick test
    n_phones = 220  # typical AISHELL phoneme vocab
    model = build_model(n_phones)

    B, T, C = 4, 300, 40
    mel = torch.randn(B, T, C)
    lengths = torch.tensor([300, 280, 250, 200])

    log_probs, out_len = model(mel, lengths)
    print(f"Input:     [{B}, {T}, {C}]")
    print(f"Output:    {list(log_probs.shape)}  (T', B, n_phones)")
    print(f"Lengths:   {out_len.tolist()}")
    print(f"Params:    {model.count_parameters():,}")

    # Frame rate: 10ms hop × 2 stride = 20ms per output frame
    # 50ms window = 2-3 frames → sufficient for 50ms accuracy target
    print(f"\nFrame shift: 20ms per output frame (< 50ms target ✓)")
