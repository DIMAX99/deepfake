import torch
import numpy as np
import base64
import cv2
from typing import List, Tuple
from torchvision import transforms
from torch import nn

from pathlib import Path

from app.utils.cvit import CViT

class CViTPredictor:
    def __init__(self, model_path: str = None):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent / "model" / "cvit_all.pth"

        self.model = CViT(
            image_size=224,
            patch_size=7,
            num_classes=2,
            channels=512,
            dim=1024,
            depth=6,
            heads=8,
            mlp_dim=2048
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])
        
    def process_frames(self, face_frames_base64: List[str]) -> Tuple[int, float, List[dict]]:
        
        if len(face_frames_base64) == 0:
            return None, 0.0, []
        
        frames = []
        per_frame_results = []
        
        for idx, frame_base64 in enumerate(face_frames_base64):
            img_bytes = base64.b64decode(frame_base64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = self.transform(frame)
                frames.append(frame_tensor)
                per_frame_results.append({
                    'frame_index': idx,
                })
        
        if len(frames) == 0:
            return None, 0.0, []
        
        frames_tensor = torch.stack(frames).to(self.device)
        
        with torch.no_grad():
            logits = self.model(frames_tensor)
            probs = torch.sigmoid(logits)
            mean_probs = torch.mean(probs, dim=0)
            
            prediction = torch.argmax(mean_probs).item()
            confidence = mean_probs[prediction].item() * 100
        
        return prediction, confidence, per_frame_results

    def get_prediction_label(self, prediction: int) -> str:
        return "FAKE" if prediction == 0 else "REAL"