import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATH = Path("model/model.pth")

CLASS_NAMES = [
    "NORMAL",
    "PNEUMONIA"
]


# ============================================================
# CREATE MODEL
# ============================================================

def create_model():

    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    number_of_features = model.fc.in_features

    model.fc = nn.Linear(
        number_of_features,
        len(CLASS_NAMES)
    )

    return model


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Trained model not found at: "
            f"{MODEL_PATH.resolve()}"
        )

    model = create_model()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)

    model.eval()

    print("=" * 60)
    print("MEDICAL AI MODEL LOADED")
    print("=" * 60)

    print("Model       : ResNet18")
    print("Device      :", DEVICE)
    print("Model path  :", MODEL_PATH.resolve())
    print("Classes     :", CLASS_NAMES)

    return model