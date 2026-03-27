"""
ONNX export utilities for fraud detection models.

Enables deployment in production environments.
"""

from pathlib import Path
from typing import Tuple

import torch
import torch.onnx


def export_lstm_to_onnx(
    model: torch.nn.Module,
    max_sequence_length: int,
    output_path: str,
    input_dim: int = 20,
    opset_version: int = 17,
    dynamic_axes: bool = True
):
    """
    Export LSTM model to ONNX format.

    Args:
        model: Trained LSTMAttentionClassifier model
        max_sequence_length: Maximum sequence length
        output_path: Path to save ONNX model
        input_dim: Number of input features
        opset_version: ONNX opset version
        dynamic_axes: Whether to use dynamic axes for variable batch/seq length
    """
    model.eval()

    # Create dummy input
    # Shape: (batch_size=1, max_seq_len, input_dim)
    dummy_sequences = torch.randn(1, max_sequence_length, input_dim)
    dummy_lengths = torch.tensor([max_sequence_length])

    # Define dynamic axes for variable-length inputs
    if dynamic_axes:
        dynamic_axes_dict = {
            'padded_sequences': {0: 'batch_size', 1: 'sequence_length'},
            'lengths': {0: 'batch_size'},
            'predictions': {0: 'batch_size'}
        }
    else:
        dynamic_axes_dict = None

    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_sequences, dummy_lengths),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['padded_sequences', 'lengths'],
        output_names=['predictions'],
        dynamic_axes=dynamic_axes_dict,
        verbose=False
    )

    print(f"✓ LSTM model exported to {output_path}")


def export_baseline_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_dim: int = 20,
    opset_version: int = 17,
    dynamic_axes: bool = True
):
    """
    Export baseline model to ONNX format.

    Args:
        model: Trained BaselineMLP model
        output_path: Path to save ONNX model
        input_dim: Number of input features
        opset_version: ONNX opset version
        dynamic_axes: Whether to use dynamic axes for variable batch size
    """
    model.eval()

    # Create dummy input
    # Shape: (batch_size=1, input_dim)
    dummy_input = torch.randn(1, input_dim)

    # Define dynamic axes
    if dynamic_axes:
        dynamic_axes_dict = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    else:
        dynamic_axes_dict = None

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes_dict,
        verbose=False
    )

    print(f"✓ Baseline model exported to {output_path}")


def validate_onnx_model(
    onnx_path: str,
    sample_input: Tuple[torch.Tensor, ...],
    model_type: str = "lstm"
):
    """
    Validate ONNX model by comparing outputs.

    Args:
        onnx_path: Path to ONNX model
        sample_input: Sample input tensor(s)
        model_type: "lstm" or "baseline"
    """
    try:
        import onnx
        from onnx import checker, helper
        import onnxruntime as ort
    except ImportError:
        print("⚠ ONNX validation requires onnx and onnxruntime packages")
        print("  Install with: pip install onnx onnxruntime")
        return

    # Load and check ONNX model
    onnx_model = onnx.load(onnx_path)
    checker.check_model(onnx_model)

    print(f"✓ ONNX model at {onnx_path} is valid")

    # Print input/output info
    print("\nModel Inputs:")
    for input_tensor in onnx_model.graph.input:
        print(f"  - {input_tensor.name}: {helper.printable_type(input_tensor.type)}")

    print("\nModel Outputs:")
    for output_tensor in onnx_model.graph.output:
        print(f"  - {output_tensor.name}: {helper.printable_type(output_tensor.type)}")

    # Run inference with ONNX Runtime
    print("\nRunning inference test...")
    session = ort.InferenceSession(onnx_path)

    if model_type == "lstm":
        sequences, lengths = sample_input
        sequences_np = sequences.cpu().numpy()
        lengths_np = lengths.cpu().numpy()

        onnx_output = session.run(
            None,
            {'padded_sequences': sequences_np, 'lengths': lengths_np}
        )
    else:  # baseline
        input_tensor = sample_input[0]
        input_np = input_tensor.cpu().numpy()

        onnx_output = session.run(
            None,
            {'input': input_np}
        )

    print(f"✓ ONNX Runtime inference successful")
    print(f"  Output shape: {onnx_output[0].shape}")
    print(f"  Output range: [{onnx_output[0].min():.4f}, {onnx_output[0].max():.4f}]")


def export_model(
    model: torch.nn.Module,
    model_type: str,
    config: dict,
    output_dir: str = "onnx_models",
    validate: bool = True
):
    """
    Main export function with validation.

    Args:
        model: PyTorch model to export
        model_type: "lstm" or "baseline"
        config: Configuration dictionary
        output_dir: Directory to save ONNX models
        validate: Whether to validate the exported model
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if model_type == "lstm":
        onnx_path = output_path / "lstm_fraud_detector.onnx"
        export_lstm_to_onnx(
            model,
            max_sequence_length=config['sequence']['max_sequence_length'],
            input_dim=config['model']['input_dim'],
            output_path=str(onnx_path),
            opset_version=config['onnx']['opset_version']
        )

        if validate:
            # Create sample input for validation
            dummy_sequences = torch.randn(2, config['sequence']['max_sequence_length'], config['model']['input_dim'])
            dummy_lengths = torch.tensor([config['sequence']['max_sequence_length'], config['sequence']['max_sequence_length'] - 2])
            validate_onnx_model(str(onnx_path), (dummy_sequences, dummy_lengths), model_type="lstm")

    elif model_type == "baseline":
        onnx_path = output_path / "baseline_fraud_detector.onnx"
        export_baseline_to_onnx(
            model,
            input_dim=config['model']['input_dim'],
            output_path=str(onnx_path),
            opset_version=config['onnx']['opset_version']
        )

        if validate:
            # Create sample input for validation
            dummy_input = torch.randn(2, config['model']['input_dim'])
            validate_onnx_model(str(onnx_path), (dummy_input,), model_type="baseline")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
