import cv2
import numpy as np
import mediapipe as mp
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from config import settings
from skin_model.inference import predict_skin_scores
import mediapipe as mp

router = APIRouter()

mp_face_detection = mp.solutions.face_detection
_face_detector = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)




class SkinResponse(BaseModel):
    acne_score: float
    wrinkle_score: float
    redness_score: float
    pore_score: float
    texture_score: float


def detect_largest_face(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Detect faces and return the largest bounding box.
    
    Args:
        image: BGR image as numpy array
        
    Returns:
        Tuple of (x, y, width, height) or None if no face detected
    """
    # Convert BGR to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = _face_detector.process(image_rgb)
    
    if not results.detections:
        return None
    
    h, w = image.shape[:2]
    largest_area = 0
    largest_box = None
    
    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        
        # Convert relative to absolute coordinates
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        box_w = int(bbox.width * w)
        box_h = int(bbox.height * h)
        
        area = box_w * box_h
        if area > largest_area:
            largest_area = area
            largest_box = (x, y, box_w, box_h)
    
    return largest_box


def crop_face_safe(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop face region with bounds checking.
    
    Args:
        image: BGR image as numpy array
        bbox: Tuple of (x, y, width, height)
        
    Returns:
        Cropped face region
    """
    h, w = image.shape[:2]
    x, y, box_w, box_h = bbox
    
    # Clamp coordinates to image bounds
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + box_w)
    y2 = min(h, y + box_h)
    
    return image[y1:y2, x1:x2]


@router.post("/predict-skin", response_model=SkinResponse)
async def predict_skin(image: UploadFile = File(...)):
    """
    Analyze skin from uploaded face image.
    
    Accepts JPEG or PNG images up to MAX_UPLOAD_SIZE_MB.
    Detects face, crops it, and returns skin scores.
    """
    # Validate content type
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Only JPEG and PNG are supported.",
        )
    
    # Read image bytes
    image_bytes = await image.read()
    
    # Validate file size
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {settings.MAX_UPLOAD_SIZE_MB}MB.",
        )
    
    # Decode image
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Failed to decode image.",
        )
    
    # Detect face
    bbox = detect_largest_face(img)
    if bbox is None:
        raise HTTPException(
            status_code=400,
            detail="No face detected in the image.",
        )
    
    # Crop face
    cropped_face = crop_face_safe(img, bbox)
    
    # Validate crop
    if cropped_face.size == 0:
        raise HTTPException(
            status_code=400,
            detail="Failed to crop face region.",
        )
    
    # Run inference
    try:
        scores = predict_skin_scores(cropped_face)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Model inference failed.",
        )
    
    return SkinResponse(**scores)
