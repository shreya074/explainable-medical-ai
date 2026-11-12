from PIL import Image
from io import BytesIO

import torch
from torchvision import transforms


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def preprocess_image(image_bytes: bytes) -> torch.Tensor:

    """
    Convert uploaded image bytes into
    a model-ready PyTorch tensor.
    """

    image = Image.open(
        BytesIO(image_bytes)
    )

    image = image.convert("RGB")

    tensor = preprocess(image)

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    return tensor