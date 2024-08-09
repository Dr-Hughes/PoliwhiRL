import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class PokemonTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.state_embedding = nn.Linear(state_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.output_layer = nn.Linear(d_model, action_dim)
        
        self.d_model = d_model

    def forward(self, states, attention_mask=None):
        # states shape: (seq_len, batch_size, state_dim)
        
        # Embed states
        x = self.state_embedding(states) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer
        if attention_mask is not None:
            x = self.transformer_encoder(x, src_key_padding_mask=attention_mask)
        else:
            x = self.transformer_encoder(x)
        
        # Get Q-values for each action
        q_values = self.output_layer(x)
        
        return q_values