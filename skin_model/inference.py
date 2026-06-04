import cv2
import numpy as np
import torch
from torchvision import transforms

from config import settings
from skin_model.model import SkinScoringModel


# ImageNet normalization transform
_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Module-level singleton: load model once
_model = SkinScoringModel()
_model = _model.to(settings.DEVICE)
_model.eval()


def predict_skin_scores(image: np.ndarray) -> dict:
    """
    Predict skin scores from an image.
    
    Args:
        image: BGR image as numpy array (from cv2)
        
    Returns:
        Dictionary with skin scores (0-1 range, 4 decimal places)
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to 224x224
    image_resized = cv2.resize(image_rgb, (224, 224))
    
    # Apply transforms (ToTensor + Normalize)
    tensor = _transform(image_resized)
    
    # Add batch dimension
    tensor = tensor.unsqueeze(0)
    
    # Move to device
    tensor = tensor.to(settings.DEVICE)
    
    # Run inference
    with torch.no_grad():
        output = _model(tensor)
    
    # Extract scores
    scores = output.squeeze().cpu().numpy()
    
    return {
        "acne_score": round(float(scores[0]), 4),
        "wrinkle_score": round(float(scores[1]), 4),
        "redness_score": round(float(scores[2]), 4),
        "pore_score": round(float(scores[3]), 4),
        "texture_score": round(float(scores[4]), 4),
    }
