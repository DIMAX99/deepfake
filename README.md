# Deepfake Detection System

A full-stack deepfake detection application using deep learning models including ResNet and Transformer-based approaches with YOLOv8 face detection.

## Features

- Video upload and processing
- Real-time deepfake detection
- Multiple detection models (ResNet + Transformer)
- Face detection using YOLOv8
- Modern React + TypeScript frontend
- FastAPI backend

## Prerequisites

- Python 3.8+
- Node.js 16+
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/DIMAX99/deepfake.git
cd deepfake
```

### 2. Download Models

Download the required model files from Google Drive:

**[Download Models Here](https://drive.google.com/drive/folders/12oBRuVUNbyozmRAz9cylrirqPSIjYlgT?usp=sharing)**

After downloading, place the model files in the following structure:

```
backend/model/
├── best_spatiotemporal_model7.pth
├── best_spatiotemporal_weights.pth
├── model_87_acc_20_frames_final_data.pt
├── model_90_acc_20_frames_FF_data.pt
└── yolov8n-face.pt
```

### 3. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install
```

## Running the Application

### Start Backend Server

```bash
cd backend
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

python -m app.main
```

The backend will run on `http://localhost:8000`

### Start Frontend Development Server

```bash
cd frontend
npm run dev
```

The frontend will run on `http://localhost:5173`

## Usage

1. Open your browser and navigate to `http://localhost:5173`
2. Upload a video file using the upload interface
3. Wait for the analysis to complete
4. View the detection results

## Project Structure

```
deepfake/
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI application
│   │   └── utils/            # Detection utilities
│   │       ├── face_detector.py
│   │       ├── resnet_predictor.py
│   │       └── transformer_predictor.py
│   ├── model/                # Model files (download separately)
│   ├── temp_uploads/         # Temporary video storage
│   └── requirements.txt      # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── services/         # API services
│   │   └── App.tsx           # Main application
│   └── package.json          # Node dependencies
└── README.md
```

## Technologies Used

### Backend
- FastAPI
- PyTorch
- OpenCV
- YOLOv8
- NumPy

### Frontend
- React
- TypeScript
- Vite
- Tailwind CSS

## API Endpoints

- `POST /api/upload` - Upload video for analysis
- `GET /api/health` - Health check endpoint

## License

MIT

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
