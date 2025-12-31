"""
Data Loading and Preprocessing for IMDB Sentiment Analysis

This module handles:
- Text preprocessing and tokenization
- Vocabulary building
- Dataset loading (IMDB)
- DataLoader creation with proper padding
"""

import os
import re
import string
from typing import List, Tuple, Optional, Dict
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# ============================================================================
# Text Preprocessing
# ============================================================================

def preprocess_text(text: str) -> str:
    """
    Basic text preprocessing for sentiment analysis.
    
    Steps:
        1. Convert to lowercase
        2. Remove HTML tags
        3. Remove URLs
        4. Keep only alphanumeric and spaces
        5. Remove extra whitespace
    
    Args:
        text: Raw input text
        
    Returns:
        Cleaned text string
    """
    # Lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization."""
    return text.split()


# ============================================================================
# Vocabulary
# ============================================================================

class Vocabulary:
    """
    Vocabulary class for mapping tokens to indices.
    
    Special tokens:
        <pad>: Padding token (index 0)
        <unk>: Unknown token (index 1)
    
    Args:
        max_size (int): Maximum vocabulary size
        min_freq (int): Minimum frequency for a token to be included
    """
    
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    
    def __init__(self, max_size: int = 25000, min_freq: int = 2):
        self.max_size = max_size
        self.min_freq = min_freq
        
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}
        self.token_freq: Counter = Counter()
        
        # Add special tokens
        self._add_token(self.PAD_TOKEN)
        self._add_token(self.UNK_TOKEN)
        
    def _add_token(self, token: str) -> int:
        """Add a token to vocabulary."""
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token
        return self.token2idx[token]
    
    def build_vocab(self, texts: List[List[str]]) -> None:
        """
        Build vocabulary from tokenized texts.
        
        Args:
            texts: List of tokenized documents (list of token lists)
        """
        # Count token frequencies
        for tokens in texts:
            self.token_freq.update(tokens)
        
        # Add tokens meeting frequency threshold
        for token, freq in self.token_freq.most_common(self.max_size - 2):
            if freq >= self.min_freq:
                self._add_token(token)
                
    def encode(self, tokens: List[str]) -> List[int]:
        """Convert tokens to indices."""
        return [self.token2idx.get(t, self.token2idx[self.UNK_TOKEN]) for t in tokens]
    
    def decode(self, indices: List[int]) -> List[str]:
        """Convert indices to tokens."""
        return [self.idx2token.get(i, self.UNK_TOKEN) for i in indices]
    
    @property
    def pad_idx(self) -> int:
        return self.token2idx[self.PAD_TOKEN]
    
    @property
    def unk_idx(self) -> int:
        return self.token2idx[self.UNK_TOKEN]
    
    def __len__(self) -> int:
        return len(self.token2idx)
    
    def save(self, path: str) -> None:
        """Save vocabulary to file."""
        torch.save({
            'token2idx': self.token2idx,
            'idx2token': self.idx2token,
            'max_size': self.max_size,
            'min_freq': self.min_freq
        }, path)
        
    @classmethod
    def load(cls, path: str) -> 'Vocabulary':
        """Load vocabulary from file."""
        data = torch.load(path)
        vocab = cls(max_size=data['max_size'], min_freq=data['min_freq'])
        vocab.token2idx = data['token2idx']
        vocab.idx2token = data['idx2token']
        return vocab


# ============================================================================
# Dataset
# ============================================================================

class SentimentDataset(Dataset):
    """
    PyTorch Dataset for sentiment analysis.
    
    Args:
        texts: List of tokenized texts
        labels: List of labels (0 or 1)
        vocab: Vocabulary object
        max_len: Maximum sequence length (truncate longer sequences)
    """
    
    def __init__(
        self,
        texts: List[List[str]],
        labels: List[int],
        vocab: Vocabulary,
        max_len: int = 256
    ):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        tokens = self.texts[idx][:self.max_len]
        label = self.labels[idx]
        
        # Encode tokens
        indices = self.vocab.encode(tokens)
        
        return torch.tensor(indices, dtype=torch.long), label, len(indices)


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for DataLoader.
    
    Pads sequences to the same length within a batch.
    
    Args:
        batch: List of (text_tensor, label, length) tuples
        
    Returns:
        texts: Padded text tensor (batch_size, max_seq_len)
        labels: Label tensor (batch_size,)
        lengths: Original lengths tensor (batch_size,)
    """
    texts, labels, lengths = zip(*batch)
    
    # Pad sequences
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    
    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    return texts_padded, labels, lengths


# ============================================================================
# IMDB Data Module
# ============================================================================

class IMDBDataModule:
    """
    Data module for IMDB dataset.
    
    Handles downloading, preprocessing, and creating DataLoaders.
    
    Args:
        data_dir: Directory to store/load data
        batch_size: Batch size for DataLoaders
        max_vocab_size: Maximum vocabulary size
        max_seq_len: Maximum sequence length
        min_freq: Minimum token frequency for vocabulary
    """
    
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 32,
        max_vocab_size: int = 25000,
        max_seq_len: int = 256,
        min_freq: int = 2
    ):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.max_vocab_size = max_vocab_size
        self.max_seq_len = max_seq_len
        self.min_freq = min_freq
        
        self.vocab: Optional[Vocabulary] = None
        self.train_dataset: Optional[SentimentDataset] = None
        self.test_dataset: Optional[SentimentDataset] = None
        
    def prepare_data(self) -> None:
        """Download IMDB dataset if not present."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Try using torchtext
            from torchtext.datasets import IMDB
            # This will download if not present
            _ = IMDB(root=str(self.data_dir), split='train')
            print("✓ IMDB dataset ready")
        except Exception as e:
            print(f"Note: Could not auto-download IMDB: {e}")
            print("Dataset will be loaded from local files if available.")
    
    def setup(self) -> None:
        """Load and preprocess data, build vocabulary."""
        print("Loading IMDB dataset...")
        
        train_texts, train_labels = self._load_split('train')
        test_texts, test_labels = self._load_split('test')
        
        print(f"  Train samples: {len(train_texts)}")
        print(f"  Test samples: {len(test_texts)}")
        
        # Preprocess and tokenize
        print("Preprocessing text...")
        train_tokens = [tokenize(preprocess_text(t)) for t in train_texts]
        test_tokens = [tokenize(preprocess_text(t)) for t in test_texts]
        
        # Build vocabulary
        print("Building vocabulary...")
        self.vocab = Vocabulary(max_size=self.max_vocab_size, min_freq=self.min_freq)
        self.vocab.build_vocab(train_tokens)
        print(f"  Vocabulary size: {len(self.vocab)}")
        
        # Create datasets
        self.train_dataset = SentimentDataset(
            train_tokens, train_labels, self.vocab, self.max_seq_len
        )
        self.test_dataset = SentimentDataset(
            test_tokens, test_labels, self.vocab, self.max_seq_len
        )
        
    def _load_split(self, split: str) -> Tuple[List[str], List[int]]:
        """Load a data split using torchtext."""
        try:
            from torchtext.datasets import IMDB
            
            dataset = IMDB(root=str(self.data_dir), split=split)
            
            texts = []
            labels = []
            for label, text in dataset:
                texts.append(text)
                # torchtext IMDB: 1=neg, 2=pos -> convert to 0/1
                labels.append(0 if label == 1 else 1)
                
            return texts, labels
            
        except Exception as e:
            print(f"Error loading {split} split: {e}")
            # Return dummy data for testing
            return self._get_dummy_data()
    
    def _get_dummy_data(self) -> Tuple[List[str], List[int]]:
        """Generate dummy data for testing when IMDB unavailable."""
        positive = [
            "This movie was absolutely fantastic! Great acting and plot.",
            "I loved every minute of this film. Highly recommended!",
            "Amazing cinematography and stellar performances throughout.",
            "One of the best movies I've seen this year. Truly inspiring.",
            "Brilliant direction and a captivating storyline.",
        ] * 100
        
        negative = [
            "This movie was terrible. Complete waste of time.",
            "I couldn't finish watching it. So boring and predictable.",
            "Awful acting and a nonsensical plot. Avoid at all costs.",
            "One of the worst films ever made. Absolutely dreadful.",
            "Poor direction and uninteresting characters throughout.",
        ] * 100
        
        texts = positive + negative
        labels = [1] * len(positive) + [0] * len(negative)
        
        return texts, labels
    
    def train_dataloader(self) -> DataLoader:
        """Get training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True
        )


if __name__ == "__main__":
    # Test the data module
    dm = IMDBDataModule(batch_size=4)
    dm.prepare_data()
    dm.setup()
    
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    texts, labels, lengths = batch
    
    print(f"\nBatch shapes:")
    print(f"  Texts: {texts.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Lengths: {lengths}")
    
    # Decode first example
    tokens = dm.vocab.decode(texts[0].tolist())
    print(f"\nFirst example tokens: {' '.join(tokens[:20])}...")
    print(f"Label: {'Positive' if labels[0] == 1 else 'Negative'}")
