"""
LSTM Model for Sentiment Analysis

This module contains the LSTM architecture for binary sentiment classification.
Includes both a basic LSTM and an attention-enhanced version.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMSentiment(nn.Module):
    """
    Basic Bidirectional LSTM for Sentiment Analysis.
    
    Architecture:
        Embedding -> BiLSTM -> FC -> Output
    
    Args:
        vocab_size (int): Size of vocabulary
        embed_dim (int): Dimension of word embeddings
        hidden_dim (int): Hidden dimension of LSTM
        output_dim (int): Number of output classes (2 for binary)
        num_layers (int): Number of LSTM layers
        dropout (float): Dropout probability
        pad_idx (int): Padding index for embedding layer
        pretrained_embeddings (torch.Tensor, optional): Pretrained embedding weights
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        output_dim: int = 2,
        num_layers: int = 2,
        dropout: float = 0.5,
        pad_idx: int = 0,
        pretrained_embeddings: torch.Tensor = None
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx
        )
        
        # Load pretrained embeddings if provided
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            # Freeze embeddings initially (can unfreeze for fine-tuning)
            self.embedding.weight.requires_grad = True
        
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Bidirectional doubles the hidden dimension
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text: torch.Tensor, text_lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            text: Input tensor of shape (batch_size, seq_len)
            text_lengths: Original lengths before padding (optional)
            
        Returns:
            Output logits of shape (batch_size, output_dim)
        """
        # text: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(text))
        # embedded: (batch_size, seq_len, embed_dim)
        
        if text_lengths is not None:
            # Pack sequence for efficiency
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, (hidden, cell) = self.lstm(packed)
            # Unpack
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            output, (hidden, cell) = self.lstm(embedded)
        
        # hidden: (num_layers * 2, batch_size, hidden_dim)
        # Concatenate final forward and backward hidden states
        hidden_cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # hidden_cat: (batch_size, hidden_dim * 2)
        
        out = self.dropout(hidden_cat)
        out = self.fc(out)
        # out: (batch_size, output_dim)
        
        return out


class Attention(nn.Module):
    """
    Attention mechanism for LSTM outputs.
    
    Computes attention weights over LSTM hidden states and returns
    a weighted sum (context vector).
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, lstm_output: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        """
        Args:
            lstm_output: LSTM outputs of shape (batch_size, seq_len, hidden_dim * 2)
            mask: Padding mask of shape (batch_size, seq_len)
            
        Returns:
            context: Weighted context vector (batch_size, hidden_dim * 2)
            attention_weights: Attention weights (batch_size, seq_len)
        """
        # Compute attention scores
        attention_scores = self.attention(lstm_output).squeeze(-1)
        # attention_scores: (batch_size, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # Normalize with softmax
        attention_weights = F.softmax(attention_scores, dim=1)
        # attention_weights: (batch_size, seq_len)
        
        # Compute weighted sum
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)
        # context: (batch_size, hidden_dim * 2)
        
        return context, attention_weights


class LSTMAttention(nn.Module):
    """
    LSTM with Attention mechanism for Sentiment Analysis.
    
    Architecture:
        Embedding -> BiLSTM -> Attention -> FC -> Output
    
    The attention mechanism allows the model to focus on the most
    relevant words for sentiment classification.
    
    Args:
        vocab_size (int): Size of vocabulary
        embed_dim (int): Dimension of word embeddings
        hidden_dim (int): Hidden dimension of LSTM
        output_dim (int): Number of output classes
        num_layers (int): Number of LSTM layers
        dropout (float): Dropout probability
        pad_idx (int): Padding index
        pretrained_embeddings (torch.Tensor, optional): Pretrained embeddings
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        output_dim: int = 2,
        num_layers: int = 2,
        dropout: float = 0.5,
        pad_idx: int = 0,
        pretrained_embeddings: torch.Tensor = None
    ):
        super().__init__()
        
        self.pad_idx = pad_idx
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx
        )
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        text: torch.Tensor, 
        text_lengths: torch.Tensor = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with optional attention weights output.
        
        Args:
            text: Input tensor (batch_size, seq_len)
            text_lengths: Original sequence lengths
            return_attention: Whether to return attention weights
            
        Returns:
            logits: Output logits (batch_size, output_dim)
            attention_weights (optional): (batch_size, seq_len)
        """
        # Create padding mask
        mask = (text != self.pad_idx).float()
        
        embedded = self.dropout(self.embedding(text))
        
        if text_lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, (hidden, cell) = self.lstm(packed)
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            lstm_output, (hidden, cell) = self.lstm(embedded)
        
        # Apply attention
        context, attention_weights = self.attention(lstm_output, mask)
        
        out = self.dropout(context)
        out = self.fc(out)
        
        if return_attention:
            return out, attention_weights
        return out
    
    def get_attention_weights(self, text: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for visualization.
        
        Args:
            text: Input tensor (batch_size, seq_len)
            
        Returns:
            attention_weights: (batch_size, seq_len)
        """
        _, attention_weights = self.forward(text, return_attention=True)
        return attention_weights


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    vocab_size = 10000
    batch_size = 32
    seq_len = 100
    
    # Test basic LSTM
    model = LSTMSentiment(vocab_size=vocab_size)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    out = model(x)
    print(f"LSTMSentiment output shape: {out.shape}")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test LSTM with Attention
    model_attn = LSTMAttention(vocab_size=vocab_size)
    out, attn = model_attn(x, return_attention=True)
    print(f"\nLSTMAttention output shape: {out.shape}")
    print(f"Attention weights shape: {attn.shape}")
    print(f"Parameters: {count_parameters(model_attn):,}")
