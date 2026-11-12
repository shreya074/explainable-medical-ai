import cv2
import numpy as np
import torch

from PIL import Image
from io import BytesIO

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def generate_gradcam(
    model,
    image_bytes,
    input_tensor,
    target_class
):
    """
    Generate a Grad-CAM visualization for
    the predicted class.
    """

    # --------------------------------------------------------
    # Load original image
    # --------------------------------------------------------

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    # Resize for visualization
    image = image.resize(
        (224, 224)
    )

    # Convert to NumPy
    rgb_image = np.array(
        image
    ).astype(
        np.float32
    ) / 255.0


    # --------------------------------------------------------
    # ResNet18 target layer
    # --------------------------------------------------------

    target_layers = [
        model.layer4[-1]
    ]


    # --------------------------------------------------------
    # Create Grad-CAM
    # --------------------------------------------------------

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )


    # Tell Grad-CAM which class to explain
    targets = [
        ClassifierOutputTarget(
            target_class
        )
    ]


    # --------------------------------------------------------
    # Generate CAM
    # --------------------------------------------------------

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]


    # --------------------------------------------------------
    # Overlay CAM on image
    # --------------------------------------------------------

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )


    # --------------------------------------------------------
    # Convert RGB → BGR
    # --------------------------------------------------------

    visualization = cv2.cvtColor(
        visualization,
        cv2.COLOR_RGB2BGR
    )


    return visualization