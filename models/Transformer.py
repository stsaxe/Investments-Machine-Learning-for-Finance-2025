import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    Implements the positional encoding function with batch-first input.
    Input shape: (batch_size, seq_len, d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Args:
            d_model (int): the dimension of the model (embedding size)
            dropout (float): dropout probability applied after adding positional encodings
            max_len (int): maximum sequence length to precompute positional encodings for
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a (max_len, d_model) matrix where each row corresponds to the positional encoding for that position.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # shape: (max_len, 1)
        # Compute the div_term (exponential decay term based on the dimension)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply sine to even indices; cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Reshape to (1, max_len, d_model) so that it can be broadcast across the batch dimension.
        pe = pe.unsqueeze(0)

        # Register as a buffer so it is saved in the model state (but not trained)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor with shape (batch_size, seq_len, d_model)
        Returns:
            Tensor: The input tensor with the positional encodings added, then dropout applied.
        """
        # Add positional encoding to the input: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ----------------------------------------
# Transformer Model for Time Series Forecasting
# ----------------------------------------
class TimeSeriesTransformer(nn.Module):
    def __init__(self,
                 src_input_dim,  # For encoder: e.g., univariate cosine input (1)
                 tgt_input_dim,  # For decoder teacher-forcing input (1)
                 model_dim,  # internal transformer (embedding) dimension
                 num_heads,
                 num_encoder_layers,
                 num_decoder_layers,
                 output_dim,  # forecast output dimension (1 for univariate)
                 dropout=0.1,
                 fc_size = 512):
        super(TimeSeriesTransformer, self).__init__()
        # Separate projections for encoder and decoder inputs:
        self.src_linear = nn.Linear(src_input_dim, model_dim)
        self.tgt_linear = nn.Linear(tgt_input_dim, model_dim)
        # Final projection from transformer output to forecast dimension
        self.output_linear = nn.Linear(model_dim, output_dim)

        self.pos_encoder = PositionalEncoding(model_dim, dropout)

        # Instantiate the transformer; note that we will pass masks during inference.
        self.transformer = nn.Transformer(d_model=model_dim,
                                          nhead=num_heads,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward = fc_size,
                                          dropout=dropout,
                                          batch_first=True)

    def forward(self, src, tgt, tgt_mask=None):
        tgt = tgt.unsqueeze(-1)
        """
        src: tensor shape (batch, src_seq_len, src_input_dim)
        tgt: tensor shape (batch, tgt_seq_len, tgt_input_dim)
        tgt_mask: optional mask for the decoder, shape (tgt_seq_len, tgt_seq_len)
        Returns:
            Tensor of shape (batch, tgt_seq_len, output_dim)
        """
        # Project inputs into embedding space.
        src_emb = self.src_linear(src)
        tgt_emb = self.tgt_linear(tgt)

        # Add positional encodings.
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)

        # Pass through transformer with an optional decoder mask.
        output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        output = self.output_linear(output)  # shape: (batch, tgt_seq_len, output_dim)
        return output
