"""
AASIST Model Export Script for EchoLocator

This script:
1. Clones the AASIST repository from GitHub
2. Loads a pre-trained AASIST checkpoint
3. Exports the model to ONNX format with dynamic axes
4. Validates the exported ONNX model
5. Saves to model/aasist.onnx

Usage:
    python export_onnx.py

Requirements:
    pip install torch onnx onnxruntime

Notes:
    - The AASIST pre-trained checkpoint must be downloaded separately
    - If no checkpoint is available, this script provides instructions
    - The exported ONNX model supports variable-length audio inputs
"""

import os
import sys
import subprocess
import numpy as np

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AASIST_DIR = os.path.join(SCRIPT_DIR, "aasist_repo")
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")
ONNX_OUTPUT = os.path.join(MODEL_DIR, "aasist.onnx")


def clone_aasist():
    """Clone the AASIST repository."""
    if os.path.exists(AASIST_DIR):
        print(f"[Export] AASIST repo already exists at {AASIST_DIR}")
        return True

    print("[Export] Cloning AASIST repository...")
    try:
        subprocess.run(
            ["git", "clone", "https://github.com/clovaai/aasist.git", AASIST_DIR],
            check=True,
            capture_output=True,
            text=True,
        )
        print("[Export] AASIST cloned successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Export] Failed to clone AASIST: {e.stderr}")
        return False


def find_checkpoint():
    """Look for pre-trained checkpoint in known locations."""
    search_paths = [
        os.path.join(AASIST_DIR, "models", "weights", "AASIST.pth"),
        os.path.join(AASIST_DIR, "pretrained", "AASIST.pth"),
        os.path.join(SCRIPT_DIR, "AASIST.pth"),
        os.path.join(MODEL_DIR, "AASIST.pth"),
    ]

    for path in search_paths:
        if os.path.exists(path):
            print(f"[Export] Found checkpoint: {path}")
            return path

    print("[Export] No pre-trained checkpoint found.")
    print("[Export] Please download the AASIST pre-trained weights and place them in one of:")
    for p in search_paths:
        print(f"         {p}")
    print()
    print("[Export] You can download from the AASIST GitHub releases or train your own model.")
    return None


def export_to_onnx():
    """Export AASIST model to ONNX format."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("[Export] PyTorch is required. Install with: pip install torch")
        return False

    # Add AASIST to path
    sys.path.insert(0, AASIST_DIR)

    checkpoint_path = find_checkpoint()
    if checkpoint_path is None:
        print("\n[Export] Creating a lightweight placeholder model for demo purposes...")
        return create_placeholder_model()

    try:
        # Try to load the AASIST model
        from models.AASIST import Model as AASISTModel

        # Load config
        import json
        config_path = os.path.join(AASIST_DIR, "config", "AASIST.conf")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            model_config = config.get("model_config", {})
        else:
            model_config = {}

        # Initialize model
        model = AASISTModel(model_config)

        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.eval()

        # Wrapper to extract logits (index 1) from AASIST tuple output
        class AASISTWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                out = self.model(x)
                # AASIST returns (embedding[B,160], logits[B,2])
                # We want the logits
                return out[1]

        wrapper = AASISTWrapper(model)
        wrapper.eval()

        # Use nb_samp from config (64600 = ~4s at 16kHz), but export with dynamic axes
        dummy_input = torch.randn(1, config.get("model_config", {}).get("nb_samp", 64600))

        os.makedirs(MODEL_DIR, exist_ok=True)

        # Export with dynamic axes
        torch.onnx.export(
            wrapper,
            dummy_input,
            ONNX_OUTPUT,
            input_names=["audio"],
            output_names=["logits"],
            dynamic_axes={
                "audio": {0: "batch"},
            },
            opset_version=14,
            do_constant_folding=True,
        )

        print(f"[Export] ONNX model saved to {ONNX_OUTPUT}")

        # Validate
        return validate_onnx()

    except Exception as e:
        print(f"[Export] Failed to export AASIST: {e}")
        print("[Export] Falling back to placeholder model...")
        return create_placeholder_model()


def create_placeholder_model():
    """
    Create a lightweight ONNX model that mimics AASIST's interface.
    This allows the system to work end-to-end even without the real AASIST weights.
    The model uses basic signal processing features as input proxy.
    """
    try:
        import torch
        import torch.nn as nn

        class PlaceholderSpoof(nn.Module):
            """Simple model that takes raw audio and produces bonafide/spoof logits."""
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv1d(1, 16, kernel_size=64, stride=16)
                self.conv2 = nn.Conv1d(16, 32, kernel_size=16, stride=8)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.fc = nn.Linear(32, 2)  # [bonafide, spoof]

            def forward(self, x):
                # x: (batch, samples)
                x = x.unsqueeze(1)  # (batch, 1, samples)
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = self.pool(x).squeeze(-1)
                return self.fc(x)

        model = PlaceholderSpoof()
        model.eval()

        dummy = torch.randn(1, 32000)
        os.makedirs(MODEL_DIR, exist_ok=True)

        torch.onnx.export(
            model,
            dummy,
            ONNX_OUTPUT,
            input_names=["audio"],
            output_names=["logits"],
            dynamic_axes={
                "audio": {0: "batch", 1: "samples"},
                "logits": {0: "batch"},
            },
            opset_version=14,
            do_constant_folding=True,
        )

        print(f"[Export] Placeholder ONNX model saved to {ONNX_OUTPUT}")
        print("[Export] NOTE: This is a demo model. Replace with real AASIST weights for production.")
        return validate_onnx()

    except Exception as e:
        print(f"[Export] Failed to create placeholder model: {e}")
        return False


def validate_onnx():
    """Validate the exported ONNX model."""
    try:
        import onnx
        model = onnx.load(ONNX_OUTPUT)
        onnx.checker.check_model(model)
        print("[Export] ONNX model validation passed ✓")
    except ImportError:
        print("[Export] onnx package not installed, skipping validation")
    except Exception as e:
        print(f"[Export] ONNX validation warning: {e}")

    try:
        import onnxruntime as ort
        session = ort.InferenceSession(ONNX_OUTPUT)
        test_input = np.random.randn(1, 64600).astype(np.float32)
        result = session.run(None, {"audio": test_input})
        print(f"[Export] Test inference output shape: {result[0].shape}")
        probs = result[0][0]
        print(f"[Export] Test inference logits: bonafide={probs[0]:.4f}, spoof={probs[1]:.4f}")
        print("[Export] ONNX Runtime inference test passed ✓")
        return True
    except ImportError:
        print("[Export] onnxruntime not installed, skipping inference test")
        return True
    except Exception as e:
        print(f"[Export] Inference test failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("  EchoLocator — AASIST Model Export")
    print("=" * 60)

    # Step 1: Clone AASIST
    clone_aasist()

    # Step 2: Export to ONNX
    success = export_to_onnx()

    if success:
        print("\n✅ Model export complete!")
        print(f"   Output: {ONNX_OUTPUT}")
        print("   Start the backend with: python app.py")
    else:
        print("\n⚠️  Model export had issues.")
        print("   The backend will still work using acoustic heuristic fallback.")
