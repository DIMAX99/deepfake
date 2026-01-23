import React, { useState } from 'react'
import { uploadMedia } from '../services/api'
import './UploadSection.css'

const UploadSection: React.FC = () => {
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState<string>('')
  const [frames, setFrames] = useState<number>(16)
  const [model, setModel] = useState<string>('resnet50')

  const models = [
    { value: 'resnet50', label: 'ResNet-50' },
    { value: 'efficientnet', label: 'EfficientNet' },
    { value: 'xception', label: 'Xception' },
    { value: 'inception', label: 'InceptionV3' },
  ]

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0]
      const fileType = selectedFile.type
      
      if (!fileType.startsWith('image/') && !fileType.startsWith('video/')) {
        setError('Please select an image or video file')
        return
      }
      
      setFile(selectedFile)
      setResult(null)
      setError('')
    }
  }

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first')
      return
    }

    setLoading(true)
    setError('')
    
    try {
      const response = await uploadMedia(file, model)
      setResult(response)
    } catch (err: any) {
      setError(err.message || 'Failed to analyze media')
    } finally {
      setLoading(false)
    }
  }

  const isVideo = file?.type.startsWith('video/')

  return (
    <div className="upload-section">
      <div className="upload-card">
        <h2 className="upload-title">Upload Media for Analysis</h2>
        <p className="upload-description">
          Upload an image or video to detect if it's a deepfake
        </p>

        <div className="upload-area">
          <input
            type="file"
            id="file-input"
            accept="image/*,video/*"
            onChange={handleFileChange}
            className="file-input"
          />
          <label htmlFor="file-input" className="file-label">
            <span className="upload-icon">🎬</span>
            <span className="upload-text">
              {file ? file.name : 'Choose an image or video file'}
            </span>
            <span className="upload-hint">Supports: JPG, PNG, MP4, AVI, MOV</span>
          </label>
        </div>

        {isVideo && (
          <div className="settings-section">
            <div className="setting-group">
              <label htmlFor="frames-input" className="setting-label">
                Number of Frames to Extract: {frames}
              </label>
              <input
                type="range"
                id="frames-input"
                min="8"
                max="50"
                value={frames}
                onChange={(e) => setFrames(Number(e.target.value))}
                className="slider"
              />
              <div className="slider-labels">
                <span>8</span>
                <span>50</span>
              </div>
            </div>
          </div>
        )}

        <div className="settings-section">
          <div className="setting-group">
            <label htmlFor="model-select" className="setting-label">
              Detection Model
            </label>
            <select
              id="model-select"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="model-select"
            >
              {models.map((m) => (
                <option key={m.value} value={m.value}>
                  {m.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        <button
          onClick={handleUpload}
          disabled={!file || loading}
          className="analyze-button"
        >
          {loading ? (
            <>
              <span className="spinner"></span>
              Analyzing...
            </>
          ) : (
            'Analyze Media'
          )}
        </button>

        {error && (
          <div className="error-message">
            ❌ {error}
          </div>
        )}

        {result && (
          <div className="result-card">
            <h3>Analysis Result</h3>
            <pre>{JSON.stringify(result, null, 2)}</pre>
          </div>
        )}
      </div>
    </div>
  )
}

export default UploadSection
