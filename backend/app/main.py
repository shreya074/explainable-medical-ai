from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from io import BytesIO
import base64

import cv2
import torch

from PIL import Image

from services.preprocessing import preprocess_image

from services.model import (
    load_model,
    DEVICE,
    CLASS_NAMES
)

from services.gradcam import generate_gradcam


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Explainable Medical AI",
    description=(
        "Academic prototype for explainable "
        "chest X-ray classification"
    ),
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# DIRECTORIES
# ============================================================

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD MODEL ON APPLICATION STARTUP
# ============================================================

model = load_model()


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "Explainable Medical AI Backend",
        "model": "ResNet18"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict_medical_image(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # 1. Validate filename
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="Filename is missing."
        )


    # --------------------------------------------------------
    # 2. Validate extension
    # --------------------------------------------------------

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png"
    }

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Only JPG, JPEG and PNG "
                "images are allowed."
            )
        )


    # --------------------------------------------------------
    # 3. Read uploaded file
    # --------------------------------------------------------

    contents = await file.read()

    if not contents:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )


    # --------------------------------------------------------
    # 4. Validate image
    # --------------------------------------------------------

    try:

        image = Image.open(
            BytesIO(contents)
        )

        image.verify()

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image."
        )


    # --------------------------------------------------------
    # 5. Save original image
    # --------------------------------------------------------

    file_path = (
        UPLOAD_DIR /
        file.filename
    )

    with open(
        file_path,
        "wb"
    ) as buffer:

        buffer.write(contents)


    # ========================================================
    # 6. PREPROCESSING
    # ========================================================

    try:

        image_tensor = preprocess_image(
            contents
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Image preprocessing failed: {str(e)}"
            )
        )


    # Move tensor to selected device
    image_tensor = image_tensor.to(
        DEVICE
    )


    # ========================================================
    # 7. AI PREDICTION
    # ========================================================

    try:

        with torch.no_grad():

            outputs = model(
                image_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted_class = torch.max(
                probabilities,
                dim=1
            )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Model prediction failed: {str(e)}"
            )
        )


    # --------------------------------------------------------
    # 8. Prediction result
    # --------------------------------------------------------

    class_index = predicted_class.item()

    prediction = CLASS_NAMES[
        class_index
    ]

    confidence_value = confidence.item()

    confidence_percentage = (
        confidence_value * 100
    )


    # ========================================================
    # 9. CONFIDENCE LEVEL
    # ========================================================

    if confidence_value >= 0.80:

        confidence_level = "HIGH"

    elif confidence_value >= 0.60:

        confidence_level = "MEDIUM"

    else:

        confidence_level = "LOW"


    # ========================================================
    # 10. GRAD-CAM
    # ========================================================

    try:

        gradcam_image = generate_gradcam(
            model=model,
            image_bytes=contents,
            input_tensor=image_tensor,
            target_class=class_index
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Grad-CAM generation failed: {str(e)}"
            )
        )


    # ========================================================
    # 11. SAVE GRAD-CAM IMAGE
    # ========================================================

    output_path = (
        OUTPUT_DIR /
        "gradcam_result.png"
    )

    cv2.imwrite(
        str(output_path),
        gradcam_image
    )


    # ========================================================
    # 12. ENCODE HEATMAP
    # ========================================================

    success, encoded_image = cv2.imencode(
        ".png",
        gradcam_image
    )

    if not success:

        raise HTTPException(
            status_code=500,
            detail="Failed to encode Grad-CAM image."
        )


    heatmap_base64 = base64.b64encode(
        encoded_image.tobytes()
    ).decode("utf-8")


    # ========================================================
    # 13. FINAL API RESPONSE
    # ========================================================

    return {

        "status": "success",

        "prediction": prediction,

        "confidence": round(
            confidence_percentage,
            2
        ),

        "confidence_level": confidence_level,

        "class_index": class_index,

        "explanation": {

            "method": "Grad-CAM",

            "description": (
                "Visualization of image regions "
                "that contributed to the model prediction."
            ),

            "heatmap": heatmap_base64
        },

        "disclaimer": (
            "This is an academic AI prototype "
            "and is not intended for clinical diagnosis."
        )
    }