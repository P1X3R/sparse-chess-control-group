# Sparse Chess Control Group Script

This is the mentioned script in [Sparse Chess](https://github.com/P1X3R/sparse-chess) that trained over the very same dataset, absolutely destroyed the Sparse Predictive Hierarchy even it's much smaller.

## Architecture

Ss a small, plain convolutional network with a shared body and two linear heads. It has no residual connections, pooling, or attention.

**Input:** `[B, 18, 8, 8]` planes decoded from the Lc0 training records:
- 12 piece planes (bitboards unpacked into 8×8 binary planes)
- 4 castling-rights planes (us O-O, us O-O-O, them O-O, them O-O-O)
- 1 side-to-move plane
- 1 rule-50 plane (`rule50_count / 100`)

**Body:** three blocks of `Conv2d(3×3, padding=1, bias=False)` -> normalization -> `ReLU`, followed by `Flatten`.
- Channels: 18 → 32 → 32 → 32 (default `channels=32`)
- Normalization: `BatchNorm2d` by default, or `GroupNorm(4, channels)`
- Output: a flat vector of size `8 × 8 × 32 = 2048`

**Heads** (both are a single `Linear` layer on the flattened features):

| Head | Layer | Output |
|------|-------|--------|
| Policy | `Linear(2048, 1858)` | Move logits (Lc0 move encoding) |
| Value | `Linear(2048, 3)` | Win/Draw/Loss logits |

**Targets and loss:**
- Policy: the MCTS visit distribution from the record. Illegal moves (marked with negative probabilities) are masked to `-1e9` before the loss.
- Value: WDL blended from the search root and the game result, `(1 - λ) · root_wdl + λ · result_wdl` with `λ = 0.25`.
- Both use `CrossEntropyLoss`. Total loss = `policy_loss + 1.0 · value_loss`.

**Optimization:** AdamW (`lr=1e-3`, `weight_decay=1e-4`), batch size 256, 1 epoch.

**Size:** about 3.84M parameters, over 99% of them in the policy head's fully connected layer.

## How to replicate?

If you wanna replicate the experiment, you first have to download one of the chunks of the [Lc0 Training Data Files](https://storage.lczero.org/files/training_data/), I specifically used one of the ***test80*** in this experiment.

### How to separate the dataset?

Just split the `.gz` files into `dataset/val/` and `dataset/train`. Since the paths are hardcoded into the code, you have to change the paths manually in the `main.py` file.

### How to setup the MLFlow server?

```bash
mlflow server --host 0.0.0.0 --port 5000
```

> [!NOTE]
> MLFlow server address is hardcoded into the `main.py`, so you have to either modify it or open the server at `http://localhost:5000`

### How to run it?

```
uv run main.py
```
