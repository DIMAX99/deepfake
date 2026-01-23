import torch
import torchvision
from torchvision import transforms
from torch import nn
from torchvision import models
import numpy as np
import cv2
import base64
from typing import List, Tuple
from pathlib import Path

class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(weights='DEFAULT')
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))


class ResNetPredictor:
    def __init__(self, model_path: str = None):
        if model_path is None:
            # Use absolute path from project root
            model_path = Path(__file__).parent.parent.parent / "model" / "model_87_acc_20_frames_final_data.pt"
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Model(2).to(self.device)
        
        # Load model weights with weights_only=False
        if Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
            print(f"ResNet model loaded successfully from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        self.model.eval()
        
        self.im_size = 112
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.sm = nn.Softmax(dim=1)
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.im_size, self.im_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
    
    def process_frames(self, face_frames_base64: List[str]) -> Tuple[int, float, List[dict]]:
        """
        Process face frames and predict if deepfake
        
        Args:
            face_frames_base64: List of base64 encoded face images
        
        Returns:
            Tuple of (prediction, confidence, per_frame_results)
            prediction: 0 for FAKE, 1 for REAL
            confidence: confidence percentage
            per_frame_results: list of results for each frame
        """
        if len(face_frames_base64) == 0:
            return None, 0.0, []
        
        # Convert base64 frames to tensors
        frames = []
        per_frame_results = []
        
        for idx, frame_base64 in enumerate(face_frames_base64):
            # Decode base64 to image
            img_bytes = base64.b64decode(frame_base64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # Transform frame
                frame_tensor = self.transform(frame)
                frames.append(frame_tensor)
        
        if len(frames) == 0:
            return None, 0.0, []
        
        # Pad to 20 frames if less
        while len(frames) < 20:
            frames.append(frames[-1])  # Repeat last frame
        
        # Take only first 20 frames if more
        frames = frames[:20]
        
        # Stack frames and add batch dimension
        frames_tensor = torch.stack(frames).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            fmap, logits = self.model(frames_tensor)
            probabilities = self.sm(logits)
            _, prediction = torch.max(probabilities, 1)
            
            confidence = probabilities[0, int(prediction.item())].item() * 100
            
            # Get per-frame analysis
            for idx in range(len(face_frames_base64)):
                frame_prob = probabilities[0].cpu().numpy()  # Same for ALL frames
                per_frame_results.append({
                    'frame_index': idx,
                    'fake_probability': float(frame_prob[0]) * 100,  # Same value repeated
                    'real_probability': float(frame_prob[1]) * 100   # Same value repeated
                })
        
        return int(prediction.item()), confidence, per_frame_results
    
    def get_prediction_label(self, prediction: int) -> str:
        """Convert prediction to label"""
        return "REAL" if prediction == 1 else "FAKE"
