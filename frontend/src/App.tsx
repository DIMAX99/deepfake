import React, { useState } from 'react';
import { Upload, Film, Settings, Play, Shield, ChevronDown, CheckCircle, AlertTriangle, Lock, Zap } from 'lucide-react';
import { uploadMedia } from './services/api';

export default function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedModel, setSelectedModel] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string>('');

  const models = [
    { id: 'resnet', name: 'ResNet-50' },
    { id: 'transformer', name: 'Swin Transformer' },
    { id: 'hybrid', name: 'Ensemble Model' }
  ];

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('video/')) {
      setSelectedFile(file);
      setError('');
      setResult(null);
    } else {
      setError('Please select a valid video file');
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) {
      setSelectedFile(file);
      setError('');
      setResult(null);
    } else {
      setError('Please drop a valid video file');
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('Please upload a video file');
      return;
    }
    if (!selectedModel) {
      setError('Please select a detection model');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await uploadMedia(selectedFile, selectedModel);
      setResult(response);
    } catch (err: any) {
      setError(err.message || 'Failed to analyze video');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 flex flex-col">
      {/* Header Bar */}
      <div className="bg-slate-800 border-b border-blue-900">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Shield className="w-7 h-7 text-blue-400" />
            <h1 className="text-xl font-semibold text-white">Deepfake</h1>
          </div>
          <div className="flex items-center gap-6 text-sm">
            <a href="#" className="text-gray-300 hover:text-white">Documentation</a>
            <a href="#" className="text-gray-300 hover:text-white">Contact</a>
          </div>
        </div>
      </div>

      <div className="flex-grow">
        <div className="max-w-6xl mx-auto px-6 py-12">
          
          {/* Hero Section */}
          <div className="mb-12 text-center">
            <h2 className="text-4xl font-bold text-white mb-4">
              AI-Powered Deepfake Detection
            </h2>
            <p className="text-lg text-gray-300 max-w-2xl mx-auto">
              Detect manipulated media. Analyze videos for deepfake manipulation in real-time.
            </p>
          </div>

          {/* Trust Banner */}
          {/* <div className="bg-blue-950 border border-blue-800 rounded-lg p-6 mb-8 flex items-start gap-4">
            <img 
              src="https://images.unsplash.com/photo-1550751827-4bd374c3f58b?w=100&h=100&fit=crop" 
              alt="AI Technology"
              className="w-20 h-20 rounded object-cover"
            />
            <div>
              <div className="flex items-center gap-2 mb-2">
                <Shield className="w-5 h-5 text-blue-400" />
                <h3 className="font-semibold text-white">Enterprise-Grade Detection</h3>
              </div>
              <p className="text-sm text-gray-300">
                Our system employs multiple neural networks trained on millions of video samples. Trusted by security agencies, media organizations, and forensic investigators worldwide.
              </p>
            </div>
          </div> */}

          {/* Main Analysis Panel */}
          <div className="bg-slate-800 border border-slate-700 rounded-lg p-8 mb-8">
            
            {/* Upload Area */}
            <div
              className={`border-2 border-dashed rounded-lg p-10 mb-8 transition-colors ${
                isDragging
                  ? 'border-blue-500 bg-blue-950'
                  : selectedFile
                  ? 'border-green-500 bg-green-950'
                  : 'border-slate-600 hover:border-slate-500 bg-slate-900'
              }`}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
            >
              {selectedFile ? (
                <div className="text-center">
                  <Film className="w-12 h-12 text-green-400 mx-auto mb-3" />
                  <p className="text-white font-medium mb-1">{selectedFile.name}</p>
                  <p className="text-gray-400 text-sm mb-4">
                    {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                  </p>
                  <button
                    onClick={() => setSelectedFile(null)}
                    className="text-blue-400 hover:text-blue-300 text-sm font-medium"
                  >
                    Remove file
                  </button>
                </div>
              ) : (
                <div className="text-center">
                  <Upload className="w-12 h-12 text-gray-500 mx-auto mb-3" />
                  <p className="text-white mb-1">
                    Drop video file here or click to upload
                  </p>
                  <p className="text-gray-400 text-sm mb-4">
                    Supports MP4, AVI, MOV, MKV up to 500MB
                  </p>
                  <label>
                    <input
                      type="file"
                      accept="video/*"
                      onChange={handleFileSelect}
                      className="hidden"
                    />
                    <span className="inline-block px-5 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded cursor-pointer">
                      Select File
                    </span>
                  </label>
                </div>
              )}
            </div>

            {/* Configuration Panel */}
            <div className="grid md:grid-cols-1 gap-6 mb-8">

              {/* Model Selection */}
              <div className="bg-slate-900 border border-slate-700 rounded-lg p-5">
                <label className="flex items-center gap-2 text-white font-medium mb-3">
                  <Zap className="w-5 h-5 text-blue-400" />
                  Detection Model
                </label>
                <div className="relative">
                  <select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded text-white appearance-none cursor-pointer hover:border-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="">Choose detection algorithm</option>
                    {models.map((model) => (
                      <option key={model.id} value={model.id}>
                        {model.name}
                      </option>
                    ))}
                  </select>
                  <ChevronDown className="w-5 h-5 text-gray-400 absolute right-3 top-2.5 pointer-events-none" />
                </div>
                <p className="text-xs text-gray-400 mt-2">
                  ResNet uses 20 frames • Other models use 12 frames
                </p>
              </div>

            </div>

            {/* Analyze Button */}
            <button
              onClick={handleAnalyze}
              disabled={!selectedFile || !selectedModel || loading}
              className="w-full px-4 py-4 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 disabled:cursor-not-allowed disabled:text-gray-500 text-white font-semibold rounded transition-colors flex items-center justify-center gap-2"
            >
              <Play className="w-5 h-5" />
              {loading ? 'Analyzing...' : 'Begin Analysis'}
            </button>

            {/* Error Message */}
            {error && (
              <div className="mt-4 p-4 bg-red-950 border border-red-800 rounded-lg flex items-center gap-3">
                <AlertTriangle className="w-5 h-5 text-red-400" />
                <p className="text-red-300">{error}</p>
              </div>
            )}

            {/* Result Display */}
            {result && (
              <div className="mt-4 space-y-4">
                {/* Main Result */}
                <div className={`p-6 border rounded-lg ${
                  result.result.is_deepfake === true 
                    ? 'bg-red-950 border-red-800' 
                    : result.result.is_deepfake === false
                    ? 'bg-green-950 border-green-800'
                    : 'bg-blue-950 border-blue-800'
                }`}>
                  <h3 className="text-white font-semibold mb-3 flex items-center gap-2">
                    {result.result.is_deepfake === true ? (
                      <>
                        <AlertTriangle className="w-5 h-5 text-red-400" />
                        DEEPFAKE DETECTED
                      </>
                    ) : result.result.is_deepfake === false ? (
                      <>
                        <CheckCircle className="w-5 h-5 text-green-400" />
                        AUTHENTIC VIDEO
                      </>
                    ) : (
                      <>
                        <CheckCircle className="w-5 h-5 text-blue-400" />
                        Analysis Complete
                      </>
                    )}
                  </h3>
                  
                  {result.result.confidence > 0 && (
                    <div className="mb-4">
                      <div className="flex justify-between text-sm text-white mb-2">
                        <span>Confidence</span>
                        <span className="font-bold">{result.result.confidence}%</span>
                      </div>
                      <div className="w-full bg-slate-700 rounded-full h-3">
                        <div 
                          className={`h-3 rounded-full ${
                            result.result.is_deepfake ? 'bg-red-500' : 'bg-green-500'
                          }`}
                          style={{ width: `${result.result.confidence}%` }}
                        ></div>
                      </div>
                    </div>
                  )}
                  
                  <p className="text-gray-300 text-sm">{result.result.note}</p>
                </div>
                
                {/* Extracted Faces */}
                {result.face_frames && result.face_frames.length > 0 && (
                  <div className="p-6 bg-slate-800 border border-slate-700 rounded-lg">
                    <h4 className="text-white font-medium mb-3">
                      Extracted Face Frames ({result.face_frames.length})
                    </h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-3">
                      {result.face_frames.map((frame: string, index: number) => (
                        <div key={index} className="border border-slate-600 rounded-lg overflow-hidden">
                          <img 
                            src={`data:image/jpeg;base64,${frame}`}
                            alt={`Face ${index + 1}`}
                            className="w-full h-auto"
                          />
                          <div className="bg-slate-900 p-2 text-center">
                            <span className="text-xs text-gray-400">Frame {index + 1}</span>
                            {result.per_frame_analysis && result.per_frame_analysis[index] && (
                              <div className="text-xs mt-1">
                                <span className={`${
                                  result.per_frame_analysis[index].real_probability > 50 
                                    ? 'text-green-400' 
                                    : 'text-red-400'
                                }`}>
                                  {result.per_frame_analysis[index].real_probability > 50 ? 'Real' : 'Fake'}: {' '}
                                  {Math.max(
                                    result.per_frame_analysis[index].real_probability,
                                    result.per_frame_analysis[index].fake_probability
                                  ).toFixed(1)}%
                                </span>
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                {/* Analysis Details */}
                <div className="bg-slate-800 border border-slate-700 rounded-lg p-4">
                  <h4 className="text-white font-medium mb-2">Analysis Details</h4>
                  <pre className="text-sm text-gray-300 overflow-x-auto">
                    {JSON.stringify({
                      file_info: result.file_info,
                      analysis_params: result.analysis_params
                    }, null, 2)}
                  </pre>
                </div>
              </div>
            )}
          </div>

          {/* Features Grid */}
          {/* <div className="grid md:grid-cols-3 gap-6 mb-8">
            <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
              <div className="flex items-center gap-3 mb-3">
                <img 
                  src="https://images.unsplash.com/photo-1677442136019-21780ecad995?w=80&h=80&fit=crop" 
                  alt="AI Analysis"
                  className="w-12 h-12 rounded object-cover"
                />
                <CheckCircle className="w-6 h-6 text-green-400" />
              </div>
              <h3 className="font-semibold text-white mb-2">99.2% Accuracy</h3>
              <p className="text-sm text-gray-400">
                State-of-the-art deep learning models detect even sophisticated manipulation attempts
              </p>
            </div>

            <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
              <div className="flex items-center gap-3 mb-3">
                <img 
                  src="https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?w=80&h=80&fit=crop" 
                  alt="Fast Processing"
                  className="w-12 h-12 rounded object-cover"
                />
                <Zap className="w-6 h-6 text-blue-400" />
              </div>
              <h3 className="font-semibold text-white mb-2">Real-Time Analysis</h3>
              <p className="text-sm text-gray-400">
                GPU-accelerated processing delivers results in under 2 minutes
              </p>
            </div>

            <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
              <div className="flex items-center gap-3 mb-3">
                <img 
                  src="https://images.unsplash.com/photo-1614064641938-3bbee52942c7?w=80&h=80&fit=crop" 
                  alt="Secure"
                  className="w-12 h-12 rounded object-cover"
                />
                <Lock className="w-6 h-6 text-yellow-400" />
              </div>
              <h3 className="font-semibold text-white mb-2">Military-Grade Security</h3>
              <p className="text-sm text-gray-400">
                End-to-end encryption with automatic file deletion after analysis
              </p>
            </div>
          </div> */}

        </div>
      </div>

      {/* Footer */}
      <footer className="bg-slate-800 border-t border-slate-700 mt-12">
        <div className="max-w-6xl mx-auto px-6 py-8">
          <div className="grid md:grid-cols-4 gap-8 mb-6">
            <div>
              <div className="flex items-center gap-2 mb-3">
                <Shield className="w-6 h-6 text-blue-400" />
                <h3 className="font-semibold text-white">Deepfake</h3>
              </div>
              <p className="text-sm text-gray-400">
                Advanced deepfake detection for media verification and threat assessment.
              </p>
            </div>
            
            <div>
              <h4 className="font-semibold text-white mb-3">Product</h4>
              <ul className="space-y-2 text-sm text-gray-400">
                <li><a href="#" className="hover:text-blue-400">Features</a></li>
                <li><a href="#" className="hover:text-blue-400">Pricing</a></li>
                <li><a href="#" className="hover:text-blue-400">Enterprise</a></li>
                <li><a href="#" className="hover:text-blue-400">API Access</a></li>
              </ul>
            </div>

            <div>
              <h4 className="font-semibold text-white mb-3">Resources</h4>
              <ul className="space-y-2 text-sm text-gray-400">
                <li><a href="#" className="hover:text-blue-400">Documentation</a></li>
                <li><a href="#" className="hover:text-blue-400">Research</a></li>
                <li><a href="#" className="hover:text-blue-400">Blog</a></li>
                <li><a href="#" className="hover:text-blue-400">Support</a></li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-semibold text-white mb-3">Company</h4>
              <ul className="space-y-2 text-sm text-gray-400">
                <li><a href="#" className="hover:text-blue-400">About</a></li>
                <li><a href="#" className="hover:text-blue-400">Privacy</a></li>
                <li><a href="#" className="hover:text-blue-400">Terms</a></li>
                <li><a href="#" className="hover:text-blue-400">Contact</a></li>
              </ul>
            </div>
          </div>
          
          <div className="border-t border-slate-700 pt-6 flex flex-col md:flex-row justify-between items-center text-sm text-gray-500">
            <p>© 2024 Deepfake Detection. All rights reserved.</p>
            <p className="mt-2 md:mt-0">Powered by Advanced Neural Networks</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
