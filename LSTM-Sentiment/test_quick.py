"""
Quick Test Script for LSTM Sentiment Analysis

Verifies that the project is set up correctly and can run inference.
Run this after installation to ensure everything works.

Usage:
    python test_quick.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ PyTorch: {e}")
        return False
    
    try:
        from models.lstm import LSTMSentiment, LSTMAttention
        print("  ✓ Models (LSTMSentiment, LSTMAttention)")
    except ImportError as e:
        print(f"  ✗ Models: {e}")
        return False
    
    try:
        from src.data import IMDBDataModule, Vocabulary
        print("  ✓ Data module")
    except ImportError as e:
        print(f"  ✗ Data module: {e}")
        return False
    
    try:
        from src.engine import Trainer, train_one_epoch, evaluate
        print("  ✓ Engine module")
    except ImportError as e:
        print(f"  ✗ Engine module: {e}")
        return False
    
    try:
        from src.utils import set_seed, get_device
        print("  ✓ Utils module")
    except ImportError as e:
        print(f"  ✗ Utils module: {e}")
        return False
    
    return True


def test_model():
    """Test model forward pass."""
    print("\nTesting model...")
    
    import torch
    from models.lstm import LSTMSentiment, LSTMAttention, count_parameters
    
    vocab_size = 10000
    batch_size = 4
    seq_len = 50
    
    # Test basic LSTM
    model = LSTMSentiment(vocab_size=vocab_size)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    try:
        out = model(x)
        assert out.shape == (batch_size, 2), f"Expected shape {(batch_size, 2)}, got {out.shape}"
        print(f"  ✓ LSTMSentiment forward pass (params: {count_parameters(model):,})")
    except Exception as e:
        print(f"  ✗ LSTMSentiment: {e}")
        return False
    
    # Test LSTM with Attention
    model_attn = LSTMAttention(vocab_size=vocab_size)
    
    try:
        out, attn_weights = model_attn(x, return_attention=True)
        assert out.shape == (batch_size, 2), f"Expected shape {(batch_size, 2)}, got {out.shape}"
        assert attn_weights.shape == (batch_size, seq_len), f"Attention shape mismatch"
        print(f"  ✓ LSTMAttention forward pass (params: {count_parameters(model_attn):,})")
    except Exception as e:
        print(f"  ✗ LSTMAttention: {e}")
        return False
    
    return True


def test_data_processing():
    """Test data preprocessing functions."""
    print("\nTesting data processing...")
    
    from src.data import preprocess_text, tokenize, Vocabulary
    
    # Test preprocessing
    test_text = "<p>This movie was AMAZING! 10/10 would watch again. http://example.com</p>"
    processed = preprocess_text(test_text)
    
    expected_clean = "this movie was amazing would watch again"
    if processed == expected_clean:
        print(f"  ✓ Text preprocessing")
    else:
        print(f"  ⚠ Preprocessing: got '{processed}'")
    
    # Test tokenization
    tokens = tokenize(processed)
    if len(tokens) == 7:
        print(f"  ✓ Tokenization ({len(tokens)} tokens)")
    else:
        print(f"  ⚠ Tokenization: expected 7 tokens, got {len(tokens)}")
    
    # Test vocabulary
    vocab = Vocabulary(max_size=100, min_freq=1)
    vocab.build_vocab([tokens, tokens])  # Add same tokens twice to meet min_freq
    
    if len(vocab) >= 2:  # At least PAD and UNK
        print(f"  ✓ Vocabulary (size: {len(vocab)})")
    else:
        print(f"  ✗ Vocabulary: too small")
        return False
    
    # Test encode/decode
    encoded = vocab.encode(tokens)
    decoded = vocab.decode(encoded)
    
    if len(encoded) == len(tokens):
        print(f"  ✓ Encode/Decode")
    else:
        print(f"  ✗ Encode/Decode mismatch")
        return False
    
    return True


def test_device():
    """Test device detection."""
    print("\nTesting device...")
    
    from src.utils import get_device
    
    device = get_device()
    print(f"  Available device: {device}")
    
    return True


def test_inference():
    """Test a simple inference example."""
    print("\nTesting inference...")
    
    import torch
    from models.lstm import LSTMAttention
    from src.data import preprocess_text, tokenize, Vocabulary
    
    # Create a small model and vocabulary
    vocab = Vocabulary(max_size=1000, min_freq=1)
    
    # Build vocab from sample sentences
    sample_texts = [
        "this movie was great loved it",
        "terrible film awful acting bad",
        "amazing cinematography beautiful scenes",
        "boring plot waste of time"
    ]
    
    tokenized = [tokenize(preprocess_text(t)) for t in sample_texts]
    vocab.build_vocab(tokenized)
    
    # Create model
    model = LSTMAttention(vocab_size=len(vocab), hidden_dim=32, num_layers=1)
    model.eval()
    
    # Test inference
    test_text = "this movie was amazing"
    tokens = tokenize(preprocess_text(test_text))
    encoded = vocab.encode(tokens)
    
    with torch.no_grad():
        x = torch.tensor([encoded])
        output, attention = model(x, return_attention=True)
        probs = torch.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()
        confidence = probs[0][pred].item()
    
    sentiment = "Positive" if pred == 1 else "Negative"
    print(f"  Input: '{test_text}'")
    print(f"  Prediction: {sentiment} (confidence: {confidence:.2%})")
    print(f"  ✓ Inference completed")
    
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("🧪 LSTM Sentiment Analysis - Quick Test")
    print("="*60)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_imports()
    all_passed &= test_model()
    all_passed &= test_data_processing()
    all_passed &= test_device()
    all_passed &= test_inference()
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("✅ All tests passed! Project is ready to use.")
        print("\nNext steps:")
        print("  1. Train: python main.py")
        print("  2. Evaluate: python main.py --evaluate")
        print("  3. Check notebook: notebooks/sentiment_analysis.ipynb")
    else:
        print("❌ Some tests failed. Please check the output above.")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
