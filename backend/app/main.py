from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import os
import shutil
from pathlib import Path
from app.utils.face_detector import FaceDetector
from app.utils.resnet_predictor import ResNetPredictor
from app.utils.transformer_predictor import TransformerPredictor

app = FastAPI(title="Deepfake Detection API")

# Initialize face detector
face_detector = FaceDetector()

# Initialize ResNet predictor
resnet_predictor = None
try:
    resnet_predictor = ResNetPredictor()
except Exception as e:
    print(f"Warning: Could not load ResNet model: {e}")

# Initialize Transformer predictor
transformer_predictor = None
try:
    transformer_predictor = TransformerPredictor()
except Exception as e:
    print(f"Warning: Could not load Transformer model: {e}")

# Create temp directory for uploads
UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",  # Vite default port
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Deepfake Detection API"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "message": "Backend is running"}

@app.post("/api/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    model: str = Form(...),
):
    """
    Endpoint to analyze uploaded video for deepfake detection
    """
    # Determine number of frames based on model type
    if 'resnet' in model.lower():
        num_frames = 20
    else:  # transformer models (xception, efficientnet, inception, ensemble)
        num_frames = 12
    
    # Save uploaded file temporarily
    temp_file_path = UPLOAD_DIR / file.filename
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get file size
        file_size = os.path.getsize(temp_file_path)
        
        # Extract faces from video
        face_frames = face_detector.extract_faces_from_video(
            str(temp_file_path),
            num_frames=num_frames,
            padding=0.2  # 20% padding
        )
        
        # Initialize result
        prediction_result = {
            "is_deepfake": None,
            "confidence": 0.0,
            "prediction_label": "Not analyzed",
            "note": "Face extraction completed"
        }
        
        per_frame_results = []
        
        # Run prediction based on selected model
        if len(face_frames) > 0:
            if 'resnet' in model.lower() and resnet_predictor is not None:
                prediction, confidence, per_frame_results = resnet_predictor.process_frames(face_frames)
                
                if prediction is not None:
                    prediction_label = resnet_predictor.get_prediction_label(prediction)
                    prediction_result = {
                        "is_deepfake": prediction == 0,  # 0 = FAKE, 1 = REAL
                        "confidence": round(confidence, 2),
                        "prediction_label": prediction_label,
                        "note": f"Video classified as {prediction_label} with {confidence:.2f}% confidence"
                    }
            
            elif transformer_predictor is not None:
                prediction, confidence, per_frame_results = transformer_predictor.process_frames(face_frames)
                
                if prediction is not None:
                    prediction_label = transformer_predictor.get_prediction_label(prediction)
                    prediction_result = {
                        "is_deepfake": prediction == 1,  # 1 = FAKE, 0 = REAL
                        "confidence": round(confidence, 2),
                        "prediction_label": prediction_label,
                        "note": f"Video classified as {prediction_label} with {confidence:.2f}% confidence"
                    }
            else:
                prediction_result["note"] = f"Model '{model}' not loaded - only face extraction performed"
        else:
            prediction_result["note"] = "No faces detected in video"
        
        return {
            "status": "success",
            "message": "Video analyzed successfully",
            "file_info": {
                "filename": file.filename,
                "content_type": file.content_type,
                "size_mb": round(file_size / (1024 * 1024), 2)
            },
            "analysis_params": {
                "frames_extracted": len(face_frames),
                "frames_used": num_frames,
                "model": model
            },
            "face_frames": face_frames,
            "result": prediction_result,
            "per_frame_analysis": per_frame_results
        }
    
    finally:
        # Clean up temporary file
        if temp_file_path.exists():
            os.remove(temp_file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
