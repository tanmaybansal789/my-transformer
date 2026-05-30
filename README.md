# Self-Attention Toy Transformer

A small character-level Transformer built from scratch in PyTorch, with rotary embeddings and causal self-attention.

## Files

- `model.py`: model + attention/rotary implementation
- `train.py`: training loop and checkpoint saving
- `generate.py`: text generation from a saved checkpoint
- `test.py`: quick checkpoint parameter-count check
- `input.txt`: training corpus

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Train (default 100 epochs):
```bash
python train.py
```

Generate text:

```bash
python generate.py
```

## Notes

- Checkpoints are saved as timestamped `.pt` files.
- Device selection is automatic: CUDA, then MPS, then CPU.
