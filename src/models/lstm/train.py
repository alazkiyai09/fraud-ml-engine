"""Pointers to preserved LSTM training assets."""


def training_entrypoints() -> list[str]:
    return [
        "scripts/lstm/train.py",
        "src/models/lstm/legacy/training/trainer.py",
    ]
