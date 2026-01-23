import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import cv2
import numpy as np
import base64
from typing import List, Tuple
from pathlib import Path

class TemporalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, T, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SpatioTemporalTransformer(nn.Module):
    def __init__(self, spatial_model='swin_tiny_patch4_window7_224', num_frames=12, 
                 num_classes=2, num_heads=8, num_temporal_layers=4, mlp_dim=2048, dropout=0.2):
        super().__init__()
        
        self.spatial_encoder = timm.create_model(
            spatial_model,
            pretrained=False,
            num_classes=0
        )
        
        self.feature_dim = self.spatial_encoder.num_features
        self.temporal_pos = nn.Parameter(torch.zeros(1, num_frames, self.feature_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_temporal_layers)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.spatial_encoder(x)
        features = features.view(B, T, -1)
        features = features + self.temporal_pos
        x = self.temporal_encoder(features)
        avg_p = torch.mean(x, dim=1)
        max_p, _ = torch.max(x, dim=1)
        return self.mlp_head(avg_p + max_p)


class TransformerPredictor:
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent / "model" / "best_spatiotemporal_weights.pth"
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Initialize model
        self.model = SpatioTemporalTransformer(
            spatial_model='swin_tiny_patch4_window7_224',
            num_frames=12,
            num_classes=2,
            num_heads=8,
            num_temporal_layers=4,
            mlp_dim=2048,
            dropout=0.2
        ).to(self.device)
        
        # Load weights only - simple and clean!
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        print(f"Transformer model loaded successfully from {model_path}")
        
        self.img_size = 224
        self.num_frames = 12
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def predict(self, frames_base64: List[str]) -> Tuple[str, float]:
        """
        Predict from base64 encoded frames
        """
        if len(frames_base64) == 0:
            return None, 0.0
        
        # Load and process frames
        loaded_frames = []
        for frame_b64 in frames_base64:
            img_bytes = base64.b64decode(frame_b64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.img_size, self.img_size))
                loaded_frames.append(frame)
        
        if len(loaded_frames) == 0:
            return None, 0.0
        
        # Pad to num_frames if less
        while len(loaded_frames) < self.num_frames:
            loaded_frames.append(loaded_frames[-1])
        
        # Take only first num_frames if more
        loaded_frames = loaded_frames[:self.num_frames]
        
        # Convert to tensors
        processed_frames = []
        for frame in loaded_frames:
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frame_tensor = self.normalize(frame_tensor)
            processed_frames.append(frame_tensor)
        
        video_tensor = torch.stack(processed_frames).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(video_tensor)
            prob = torch.softmax(output, dim=1)
            pred = output.argmax(1).item()
            confidence = prob[0][pred].item()
        
        label = 'FAKE' if pred == 1 else 'REAL'
        return label, confidence
    
    def process_frames(self, face_frames_base64: List[str]) -> Tuple[int, float, List[dict]]:
        """
        Process face frames and predict if deepfake
        """
        label, confidence = self.predict(face_frames_base64)
        
        if label is None:
            return None, 0.0, []
        
        # Convert label to prediction number
        prediction = 1 if label == 'FAKE' else 0
        confidence_pct = confidence * 100
        
        # Create per-frame results
        per_frame_results = []
        for idx in range(len(face_frames_base64)):
            per_frame_results.append({
                'frame_index': idx,
                'real_probability': (1 - confidence) * 100 if prediction == 1 else confidence * 100,  # Same for all
                'fake_probability': confidence * 100 if prediction == 1 else (1 - confidence) * 100   # Same for all
            })
        
        return prediction, confidence_pct, per_frame_results
    
    def get_prediction_label(self, prediction: int) -> str:
        """Convert prediction to label"""
        return "FAKE" if prediction == 1 else "REAL"
