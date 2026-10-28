import gzip
import random
from functools import lru_cache
from pathlib import Path

import mlflow
import mlflow.pytorch as mltorch
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

LAMBDA = 0.25

training_data_dtype = np.dtype(
    [
        ("version", "<u4"),
        ("input_format", "<u4"),
        ("probabilities", "<f4", (1858,)),
        ("planes", "<u8", (104,)),
        ("castling_us_ooo", "u1"),
        ("castling_us_oo", "u1"),
        ("castling_them_ooo", "u1"),
        ("castling_them_oo", "u1"),
        ("side_to_move_or_enpassant", "u1"),
        ("rule50_count", "u1"),
        ("invariance_info", "u1"),
        ("dummy", "u1"),
        ("root_q", "<f4"),
        ("best_q", "<f4"),
        ("root_d", "<f4"),
        ("best_d", "<f4"),
        ("root_m", "<f4"),
        ("best_m", "<f4"),
        ("plies_left", "<f4"),
        ("result_q", "<f4"),
        ("result_d", "<f4"),
        ("played_q", "<f4"),
        ("played_d", "<f4"),
        ("played_m", "<f4"),
        ("orig_q", "<f4"),
        ("orig_d", "<f4"),
        ("orig_m", "<f4"),
        ("visits", "<u4"),
        ("played_idx", "<u2"),
        ("best_idx", "<u2"),
        ("policy_kld", "<f4"),
        ("q_st", "<f4"),
    ]
)

planes_shape = (18, 8, 8)

assert training_data_dtype.itemsize == 8356, (
    f"Mismatch size: {training_data_dtype.itemsize}"
)


@lru_cache(maxsize=8)
def load_chunk_to_numpy(file_path: str | Path) -> np.ndarray:
    with gzip.open(file_path, "rb") as f:
        compressed_bytes = f.read()

    records = np.frombuffer(compressed_bytes, dtype=training_data_dtype)
    return records


BIT_MASKS = (np.uint64(1) << np.arange(64, dtype=np.uint64)).reshape(8, 8)


def record_to_planes(record):
    data_planes = record["planes"]

    planes = np.zeros(planes_shape, dtype=np.float32)

    masked = data_planes[:12, None, None] & BIT_MASKS
    planes[:12] = (masked > 0).astype(np.float32)

    turn = int(record["side_to_move_or_enpassant"] & 1)

    planes[12] = record["castling_us_oo"]
    planes[13] = record["castling_us_ooo"]
    planes[14] = record["castling_them_oo"]
    planes[15] = record["castling_them_ooo"]
    planes[16] = turn
    planes[17] = np.float32(record["rule50_count"]) / 100

    return planes


def q_d_to_win(q, d):
    return (1.0 + q - d) * 0.5


def get_wdl(q, d):
    return np.array([q_d_to_win(q, d), d, q_d_to_win(-q, d)], dtype=np.float32)


class GzipChessIterableDataset(IterableDataset):
    def __init__(
        self, data_dir: str, shuffle_chunks: bool = True, max_chunks: int | None = None
    ):
        super().__init__()
        path = Path(data_dir).expanduser()
        file_paths = sorted([str(f) for f in path.glob("*.gz")])

        if max_chunks is not None and len(file_paths) > max_chunks:
            file_paths = random.sample(file_paths, max_chunks)

        self.file_paths = file_paths
        self.shuffle_chunks = shuffle_chunks

    def __iter__(self):
        file_paths = list(self.file_paths)
        if self.shuffle_chunks:
            random.shuffle(file_paths)

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            per_worker = int(np.ceil(len(file_paths) / worker_info.num_workers))
            worker_id = worker_info.id
            file_paths = file_paths[
                worker_id * per_worker : (worker_id + 1) * per_worker
            ]

        for file_path in file_paths:
            records = load_chunk_to_numpy(file_path)

            indices = np.arange(len(records))
            np.random.shuffle(indices)

            for idx in indices:
                record = records[idx]

                probabilities = record["probabilities"]
                policy_target = np.maximum(probabilities, 0)
                legal_mask = probabilities < 0
                planes = record_to_planes(record)

                root_wdl = get_wdl(record["root_q"], record["root_d"])
                result_wdl = get_wdl(record["result_q"], record["result_d"])
                blended = (1 - LAMBDA) * root_wdl + LAMBDA * result_wdl

                yield {
                    "planes": torch.from_numpy(planes),
                    "policy": torch.from_numpy(policy_target),
                    "mask": torch.from_numpy(legal_mask),
                    "wdl": torch.tensor(blended, dtype=torch.float32),
                }


class ControlCNN(nn.Module):
    def __init__(self, channels: int = 32, norm: str = "batch"):
        super().__init__()

        def make_norm():
            return (
                nn.BatchNorm2d(channels)
                if norm == "batch"
                else nn.GroupNorm(4, channels)
            )

        self.body = nn.Sequential(
            nn.Conv2d(18, channels, 3, padding=1, bias=False),
            make_norm(),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            make_norm(),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            make_norm(),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.policy_head = nn.Linear(8 * 8 * channels, 1858)
        self.value_head = nn.Linear(8 * 8 * channels, 3)

    def forward(self, x):
        assert x.shape[1:] == planes_shape, (
            f"Expected shape [B, 18, 8, 8], got {x.shape}"
        )

        body_out = self.body(x)

        return self.policy_head(body_out), self.value_head(body_out)


def seed_everything(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        seed = worker_info.seed % (2**32)
        np.random.seed(seed)
        random.seed(seed)


VALUE_LOSS_WEIGHT = 1.0


def validate(model, device, criterion, global_step):
    model.eval()
    val_dataset = GzipChessIterableDataset("/dataset/val/", max_chunks=200)
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    val_loss = 0.0
    val_policy_loss = 0.0
    val_value_loss = 0.0
    val_steps = 0

    with torch.no_grad():
        for batch in val_loader:
            planes = batch["planes"].to(device, non_blocking=True)
            policy_target = batch["policy"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            wdl_target = batch["wdl"].to(device, non_blocking=True)

            policy_logits, value_logits = model(planes)
            policy_logits = policy_logits.masked_fill(mask, -1e9)

            policy_loss = criterion(policy_logits, policy_target)
            value_loss = criterion(value_logits, wdl_target)

            val_policy_loss += policy_loss.item()
            val_value_loss += value_loss.item()
            val_loss += policy_loss.item() + (VALUE_LOSS_WEIGHT * value_loss.item())
            val_steps += 1

    model.train()
    if val_steps > 0:
        avg_val_loss = val_loss / val_steps
        avg_val_policy = val_policy_loss / val_steps
        avg_val_value = val_value_loss / val_steps

        mlflow.log_metrics(
            {
                "val_loss": avg_val_loss,
                "val_policy_loss": avg_val_policy,
                "val_value_loss": avg_val_value,
            },
            step=global_step,
        )

        print(
            f"--- Validation Result --- "
            f"Loss [Total: {avg_val_loss:.4f}, "
            f"Policy: {avg_val_policy:.4f}, "
            f"Value: {avg_val_value:.4f}]"
        )


def train():
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("sparse-chess-control")

    with mlflow.start_run():
        learning_rate = 1e-3
        weight_decay = 1e-4
        batch_size = 256
        epochs = 1
        channels = 32

        mlflow.log_params(
            {
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "epochs": epochs,
                "channels": channels,
                "lambda": LAMBDA,
                "value_loss_weight": VALUE_LOSS_WEIGHT,
                "device": str(device),
            }
        )

        dataset = GzipChessIterableDataset("/dataset/train")
        loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=4, pin_memory=True
        )

        model = ControlCNN(channels=channels).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        global_step = 0

        model.train()
        for epoch in range(epochs):
            steps = 0
            total_loss = 0.0

            for batch_idx, batch in enumerate(loader):
                planes = batch["planes"].to(device, non_blocking=True)
                policy_target = batch["policy"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                wdl_target = batch["wdl"].to(device, non_blocking=True)

                optimizer.zero_grad()

                policy_logits, value_logits = model(planes)
                policy_logits = policy_logits.masked_fill(mask, -1e9)

                policy_loss = criterion(policy_logits, policy_target)
                value_loss = criterion(value_logits, wdl_target)

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()

                loss = policy_loss + (VALUE_LOSS_WEIGHT * value_loss)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                steps += 1
                global_step += 1

                if batch_idx % 100 == 0:
                    current_train_loss = loss.item()
                    current_policy_loss = policy_loss.item()
                    current_value_loss = value_loss.item()

                    mlflow.log_metrics(
                        {
                            "train_loss": current_train_loss,
                            "train_policy_loss": current_policy_loss,
                            "train_value_loss": current_value_loss,
                        },
                        step=global_step,
                    )

                    print(
                        f"Epoch [{epoch + 1}/{epochs}] | Step [{batch_idx}] | "
                        f"Loss [Policy: {total_policy_loss / steps:.4f}, Value: {total_value_loss / steps:.4f}]"
                    )

                if batch_idx > 0 and batch_idx % 1000 == 0:
                    validate(model, device, criterion, global_step)

            epoch_avg_loss = total_loss / steps
            mlflow.log_metric("epoch_loss", epoch_avg_loss, step=global_step)
            print(
                f"--- Epoch {epoch + 1} Complete. Average Loss: {epoch_avg_loss:.4f} ---"
            )

        sample_batch = next(iter(loader))
        input_example = sample_batch["planes"][:1].to(device)
        mltorch.log_model(
            model, name="model", input_example=input_example, serialization_format="pt2"
        )
        print("Model successfully logged to MLflow.")


if __name__ == "__main__":
    train()
