import React, { useState, useRef } from 'react';
import axios from 'axios';
import './App.css';

/**
 * Main React Component - Stress Fracture Predictor Interface
 * 
 * Features:
 * - Upload CAD files (STL/STEP)
 * - Interactive parameter sliders
 * - Real-time stress visualization
 * - Hotspot detection and display
 * - Design recommendations
 * - Export results as JSON/CSV
 */

export default function App() {
  // File and loading state
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Parameter state (what user can control)
  const [loadX, setLoadX] = useState(256);
  const [loadY, setLoadY] = useState(256);
  const [loadMagnitude, setLoadMagnitude] = useState(500);

  // Results state
  const [result, setResult] = useState(null);
  const [selectedMaterial, setSelectedMaterial] = useState('ASTM_A36');

  /**
   * Handle file upload
   */
  const handleFileUpload = (e) => {
    const uploadedFile = e.target.files[0];
    if (uploadedFile) {
      setFile(uploadedFile);
      setError(null);
    }
  };

  /**
   * Main prediction function - Send to backend API
   */
  const handlePredictClick = async (e) => {
    e.preventDefault();

    if (!file) {
      setError('Please select a CAD file');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('load_x', loadX);
    formData.append('load_y', loadY);
    formData.append('load_magnitude', loadMagnitude);

    try {
      // Call backend API
      const response = await axios.post('http://localhost:8000/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 30000
      });

      setResult(response.data);
    } catch (err) {
      const errorMsg = err.response?.data?.detail || err.message || 'Prediction failed';
      setError(errorMsg);
      console.error('API Error:', err);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Convert stress array to visualization image
   */
  const createStressImage = (stressArray) => {
    if (!stressArray) return null;

    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(512, 512);
    const data = imageData.data;

    // Flatten and normalize stress values
    const flat = stressArray.flat();
    const maxStress = Math.max(...flat);

    // Color mapping: Blue (low) -> Red (high)
    for (let i = 0; i < flat.length; i++) {
      const normalized = flat[i] / maxStress;
      const idx = i * 4;

      // Red channel: increases with stress
      data[idx] = Math.min(255, normalized * 255 * 2);
      // Green: neutral
      data[idx + 1] = Math.max(0, (1 - normalized) * 100);
      // Blue channel: decreases with stress
      data[idx + 2] = Math.max(0, (1 - normalized) * 255);
      // Alpha: always opaque
      data[idx + 3] = 255;
    }

    ctx.putImageData(imageData, 0, 0);
    return canvas.toDataURL('image/png');
  };

  /**
   * Export results as JSON
   */
  const exportJSON = () => {
    if (!result) return;
    const dataStr = JSON.stringify(result, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `stress_analysis_${Date.now()}.json`;
    link.click();
  };

  /**
   * Get risk badge color class
   */
  const getRiskColorClass = (risk) => {
    switch (risk) {
      case 'CRITICAL': return 'risk-critical';
      case 'HIGH': return 'risk-high';
      case 'MEDIUM': return 'risk-medium';
      case 'LOW': return 'risk-low';
      default: return 'risk-unknown';
    }
  };

  /**
   * Main render
   */
  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <h1>🔧 AI Stress Fracture Predictor</h1>
        <p>Neural Network-Based FEA Surrogate Model</p>
      </header>

      <main className="app-main">
        {/* Left Panel: Upload & Parameters */}
        <div className="left-panel">
          <div className="panel-section">
            <h2>📁 Upload CAD File</h2>
            <div className="file-input-wrapper">
              <input
                type="file"
                id="file-input"
                onChange={handleFileUpload}
                accept=".stl,.step,.stp"
                disabled={loading}
              />
              <label htmlFor="file-input" className="file-label">
                {file ? `✅ ${file.name}` : '📤 Choose STL/STEP file'}
              </label>
            </div>
          </div>

          {/* Parameters Section */}
          <div className="panel-section">
            <h2>⚙️ Parameters</h2>

            <div className="parameter-control">
              <label>X position: {loadX}px</label>
              <input
                type="range"
                min="0"
                max="512"
                value={loadX}
                onChange={(e) => setLoadX(parseInt(e.target.value))}
                disabled={loading}
              />
            </div>

            <div className="parameter-control">
              <label>Y position: {loadY}px</label>
              <input
                type="range"
                min="0"
                max="512"
                value={loadY}
                onChange={(e) => setLoadY(parseInt(e.target.value))}
                disabled={loading}
              />
            </div>

            <div className="parameter-control">
              <label>Load Magnitude: {loadMagnitude}N</label>
              <input
                type="range"
                min="100"
                max="1000"
                step="50"
                value={loadMagnitude}
                onChange={(e) => setLoadMagnitude(parseInt(e.target.value))}
                disabled={loading}
              />
            </div>
          </div>

          {/* Predict Button */}
          <button
            className="predict-button"
            onClick={handlePredictClick}
            disabled={!file || loading}
          >
            {loading ? '⏳ Analyzing...' : '🚀 Predict Fracture'}
          </button>

          {/* Error Display */}
          {error && (
            <div className="error-box">
              <p>❌ {error}</p>
            </div>
          )}
        </div>

        {/* Right Panel: Results */}
        <div className="right-panel">
          {!result ? (
            <div className="empty-state">
              <p>📊 Upload a CAD file and click Predict</p>
              <p>Results will appear here</p>
            </div>
          ) : (
            <>
              {/* Stress Map */}
              <div className="result-section">
                <h3>🔥 Stress Distribution</h3>
                <div className="stress-map-container">
                  {result.stress_map && (
                    <img
                      src={createStressImage(result.stress_map)}
                      alt="Stress Heatmap"
                      className="stress-map-image"
                    />
                  )}
                </div>
                <p className="stress-info">
                  Max Stress: <strong>{result.max_stress_mpa?.toFixed(1)}MPa</strong>
                </p>
              </div>

              {/* Risk Badge */}
              <div className={`risk-badge ${getRiskColorClass(result.fracture_risk)}`}>
                <h3>⚠️ Fracture Risk</h3>
                <p className="risk-text">{result.fracture_risk}</p>
                <p className="risk-info">Safety Factor: {(250 / (result.max_stress_mpa || 1)).toFixed(2)}x</p>
              </div>

              {/* Primary Hotspot */}
              {result.primary_hotspot && (
                <div className="hotspot-box">
                  <h3>🎯 Primary Hotspot</h3>
                  <p>Location: ({result.primary_hotspot.x}, {result.primary_hotspot.y})</p>
                  <p>Stress: {result.primary_hotspot.stress_mpa?.toFixed(1)}MPa</p>
                </div>
              )}

              {/* Recommendations */}
              <div className="recommendations-box">
                <h3>💡 Recommendations</h3>
                {result.recommendations && result.recommendations.length > 0 ? (
                  <ul className="recommendations-list">
                    {result.recommendations.map((rec, idx) => (
                      <li key={idx} className={`rec-item rec-${rec.priority.toLowerCase()}`}>
                        <strong>{rec.type.replace(/_/g, ' ').toUpperCase()}</strong>
                        <p>{rec.description}</p>
                        <p className="rec-impact">
                          💪 Expected improvement: {rec.expected_reduction_percent}%
                        </p>
                        <p className="rec-difficulty">Difficulty: {rec.difficulty}</p>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p>✅ No recommendations - design is safe</p>
                )}
              </div>

              {/* Export Button */}
              <button className="export-button" onClick={exportJSON}>
                📥 Export as JSON
              </button>
            </>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>AI Stress Predictor v1.0 | Powered by U-Net + CalculiX</p>
      </footer>
    </div>
  );
}
