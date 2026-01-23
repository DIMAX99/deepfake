import cv2
import numpy as np
from ultralytics import YOLO
from typing import List
import base64
from pathlib import Path

class FaceDetector:
    def __init__(self, model_name: str = None):
        """Initialize YOLO face detector"""
        if model_name is None:
            # Use absolute path from project root
            model_name = Path(__file__).parent.parent.parent / "model" / "yolov8n-face.pt"
        
        self.model = YOLO(str(model_name))
        print(f"YOLO face detector loaded from {model_name}")
    
    def extract_faces_from_video(self, video_path: str, num_frames: int = 10, padding: float = 0.2) -> List[str]:
        """
        Extract faces from video frames
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            padding: Padding around detected face (0.2 = 20%)
        
        Returns:
            List of base64 encoded cropped face images
        """
        padding=0
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return []
        
        # Calculate frame indices to sample
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        face_frames = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Detect objects using YOLO
            results = self.model(frame, verbose=False)
            
            # Process detections
            for result in results:
                boxes = result.boxes
                
                if len(boxes) == 0:
                    continue
                
                # Get the first detection (largest or most confident)
                box = boxes[0]
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                
                # Add padding
                width = x2 - x1
                height = y2 - y1
                
                pad_w = int(width * padding)
                pad_h = int(height * padding)
                
                # Calculate padded coordinates (ensuring they stay within frame bounds)
                x1_pad = max(0, x1 - pad_w)
                y1_pad = max(0, y1 - pad_h)
                x2_pad = min(frame.shape[1], x2 + pad_w)
                y2_pad = min(frame.shape[0], y2 + pad_h)
                
                # Crop ONLY the face region with padding
                face_crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                
                # Verify the crop is valid
                if face_crop.size > 0 and face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                    # Encode cropped face to JPEG
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                    _, buffer = cv2.imencode('.jpg', face_crop, encode_param)
                    
                    # Convert to base64
                    face_base64 = base64.b64encode(buffer).decode('utf-8')
                    face_frames.append(face_base64)
                    break  # Take only first/best detection per frame
        
        cap.release()
        return face_frames
