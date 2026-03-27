"""LSTM stack metadata and lazy accessors."""


def describe_lstm_stack() -> dict:
    return {
        "source": "src/models/lstm/legacy",
        "architectures": ["lstm_attention", "baseline_mlp"],
    }


def load_lstm_attention_model():
    from src.models.lstm.legacy.models.lstm_attention import LSTMAttentionClassifier

    return LSTMAttentionClassifier
