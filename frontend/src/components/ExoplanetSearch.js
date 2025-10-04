import React, { useState } from 'react';
import { Telescope, Sparkles, Database, Upload, Search, Globe } from 'lucide-react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const sampleTargets = [
  { id: 'Kepler-442b', name: 'Kepler-442b', description: 'Potentially habitable super-Earth' },
  { id: 'HD 209458b', name: 'HD 209458b', description: 'First exoplanet with detected atmosphere' },
  { id: 'TRAPPIST-1e', name: 'TRAPPIST-1e', description: 'Earth-sized planet in habitable zone' },
  { id: 'Kepler-452b', name: 'Kepler-452b', description: 'Earth\'s "cousin" orbiting a Sun-like star' },
  { id: 'Kepler-186f', name: 'Kepler-186f', description: 'First Earth-sized planet in habitable zone' },
  { id: 'Proxima Centauri b', name: 'Proxima Centauri b', description: 'Closest known exoplanet to Earth' },
];

const ExoplanetSearch = () => {
  const [selectedTarget, setSelectedTarget] = useState('');
  const [customTarget, setCustomTarget] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [uploadedFile, setUploadedFile] = useState(null);
  const [showNasaEyes, setShowNasaEyes] = useState(false);

  const analyzeTarget = async () => {
    const targetToAnalyze = customTarget.trim() || selectedTarget;

    if (!targetToAnalyze && !uploadedFile) {
      setError('Please select a target, enter a custom target, or upload a CSV file');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const formData = new FormData();
      
      if (uploadedFile) {
        formData.append('file', uploadedFile);
        formData.append('analysis_type', 'csv_batch');
      } else {
        formData.append('target_name', targetToAnalyze);
        formData.append('analysis_type', 'single_target');
      }

      const response = await axios.post(`${API}/exoplanet-analysis`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResult(response.data);
    } catch (err) {
      console.error('Analysis error:', err);
      setError(err.response?.data?.detail || err.message || 'Analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const getNasaEyesUrl = (planetName) => {
    // NASA Eyes on Exoplanets URL format
    const cleanName = planetName.replace(/\s+/g, '').toLowerCase();
    return `https://eyes.nasa.gov/apps/exo/#/${cleanName}`;
  };

  return (
    <div className="space-y-6">
      {/* Search Section */}
      <div className="bg-slate-900/70 backdrop-blur-sm rounded-xl p-8 border border-slate-800 shadow-2xl">
        <div className="flex items-center gap-2 mb-4">
          <Sparkles className="w-6 h-6 text-cyan-400" />
          <h2 className="text-xl font-bold text-white">Exoplanet Analysis</h2>
        </div>

        <div className="grid md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-slate-300 text-sm font-medium mb-2">
              Select Sample Target
            </label>
            <select
              value={selectedTarget}
              onChange={(e) => {
                setSelectedTarget(e.target.value);
                setCustomTarget('');
                setUploadedFile(null);
              }}
              className="w-full px-4 py-3 bg-slate-800 text-white rounded-lg border border-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={loading}
            >
              <option value="">Choose a target...</option>
              {sampleTargets.map((target) => (
                <option key={target.id} value={target.id}>
                  {target.name} - {target.description}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-slate-300 text-sm font-medium mb-2">
              Or Enter Custom Target
            </label>
            <input
              type="text"
              value={customTarget}
              onChange={(e) => {
                setCustomTarget(e.target.value);
                setSelectedTarget('');
                setUploadedFile(null);
              }}
              placeholder="e.g., Kepler-22b, KOI-123"
              className="w-full px-4 py-3 bg-slate-800 text-white rounded-lg border border-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={loading}
            />
          </div>
        </div>

        {/* File Upload Section */}
        <div className="mb-4">
          <label className="block text-slate-300 text-sm font-medium mb-2">
            Or Upload CSV File for Batch Analysis
          </label>
          <div className="flex items-center gap-2">
            <input
              type="file"
              accept=".csv"
              onChange={(e) => {
                const file = e.target.files[0];
                setUploadedFile(file);
                setSelectedTarget('');
                setCustomTarget('');
              }}
              className="hidden"
              id="file-upload"
              disabled={loading}
            />
            <label
              htmlFor="file-upload"
              className="flex items-center gap-2 px-4 py-3 bg-slate-800 text-white rounded-lg border border-slate-700 hover:bg-slate-700 cursor-pointer transition-colors"
            >
              <Upload className="w-4 h-4" />
              Choose CSV File
            </label>
            {uploadedFile && (
              <span className="text-slate-300 text-sm">{uploadedFile.name}</span>
            )}
          </div>
        </div>

        <button
          onClick={analyzeTarget}
          disabled={loading}
          className="w-full px-6 py-4 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white font-semibold rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              <span>Analyzing...</span>
            </>
          ) : (
            <>
              <Search className="w-5 h-5" />
              <span>Analyze Exoplanet</span>
            </>
          )}
        </button>

        {error && (
          <div className="mt-4 p-4 bg-red-900/30 border border-red-800 rounded-lg text-red-400">
            {error}
          </div>
        )}
      </div>

      {/* Results Section */}
      {result && (
        <div className="space-y-6 animate-fadeIn">
          {/* Detection Results */}
          <div className="bg-slate-900/70 backdrop-blur-sm rounded-xl p-8 border border-slate-800">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-bold text-white flex items-center gap-2">
                <Telescope className="w-6 h-6 text-cyan-400" />
                Analysis Results
              </h3>
              {result.target && (
                <button
                  onClick={() => setShowNasaEyes(!showNasaEyes)}
                  className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white rounded-lg transition-all"
                >
                  <Globe className="w-4 h-4" />
                  {showNasaEyes ? 'Hide' : 'View'} 3D Model
                </button>
              )}
            </div>

            {result.batch_results ? (
              // Batch results display
              <div>
                <h4 className="text-lg font-semibold text-white mb-4">
                  Batch Analysis Results ({result.batch_results.length} targets)
                </h4>
                <div className="space-y-4">
                  {result.batch_results.map((item, index) => (
                    <div key={index} className="p-4 bg-slate-800/50 rounded-lg">
                      <div className="grid md:grid-cols-3 gap-4">
                        <div>
                          <p className="text-slate-400 text-sm">Target</p>
                          <p className="text-white font-semibold">{item.target}</p>
                        </div>
                        <div>
                          <p className="text-slate-400 text-sm">Classification</p>
                          <p className="text-white font-semibold">{item.classification}</p>
                        </div>
                        <div>
                          <p className="text-slate-400 text-sm">Confidence</p>
                          <p className="text-white font-semibold">{(item.confidence * 100).toFixed(1)}%</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              // Single target results
              <div>
                <div className="grid md:grid-cols-3 gap-6 mb-6">
                  <div className="p-4 bg-slate-800/50 rounded-lg">
                    <p className="text-slate-400 text-sm mb-1">Target</p>
                    <p className="text-white font-semibold text-lg">{result.target}</p>
                  </div>
                  <div className="p-4 bg-slate-800/50 rounded-lg">
                    <p className="text-slate-400 text-sm mb-1">Classification</p>
                    <p className="text-white font-semibold text-lg">{result.classification}</p>
                  </div>
                  <div className="p-4 bg-slate-800/50 rounded-lg">
                    <p className="text-slate-400 text-sm mb-1">Confidence</p>
                    <p className="text-white font-semibold text-lg">{(result.confidence * 100).toFixed(1)}%</p>
                  </div>
                </div>

                {result.features && (
                  <div className="mb-6">
                    <h4 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                      <Database className="w-5 h-5" />
                      Key Features
                    </h4>
                    <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
                      {Object.entries(result.features).map(([key, value]) => (
                        <div key={key} className="p-3 bg-slate-800/50 rounded">
                          <p className="text-slate-400 text-xs mb-1">{key.replace(/_/g, ' ').toUpperCase()}</p>
                          <p className="text-white font-semibold">
                            {typeof value === 'number' ? value.toFixed(3) : value}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* NASA Eyes 3D Model Integration */}
            {showNasaEyes && result.target && (
              <div className="mt-6 p-4 bg-gradient-to-r from-purple-900/20 to-blue-900/20 rounded-lg border border-purple-500/30">
                <div className="flex items-center justify-between mb-4">
                  <h4 className="text-lg font-semibold text-white flex items-center gap-2">
                    <Globe className="w-5 h-5 text-purple-400" />
                    NASA Eyes on Exoplanets - 3D Model
                  </h4>
                  <button
                    onClick={() => window.open(getNasaEyesUrl(result.target), '_blank')}
                    className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors text-sm"
                  >
                    Open in New Tab
                  </button>
                </div>
                <div className="relative w-full h-96 bg-black rounded-lg overflow-hidden">
                  <iframe
                    src={getNasaEyesUrl(result.target)}
                    className="w-full h-full border-0"
                    title={`NASA Eyes 3D Model - ${result.target}`}
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                    allowFullScreen
                  />
                </div>
                <p className="text-slate-400 text-sm mt-2">
                  Interactive 3D visualization powered by NASA's Eyes on Exoplanets. 
                  Use mouse controls to rotate and zoom the model.
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Info Section */}
      {!result && !loading && (
        <div className="bg-slate-900/50 backdrop-blur-sm rounded-xl p-8 border border-slate-800">
          <h2 className="text-xl font-bold text-white mb-4">About This Analysis Tool</h2>
          <div className="grid md:grid-cols-2 gap-6 text-slate-300">
            <div>
              <h3 className="font-semibold text-white mb-2 flex items-center gap-2">
                <Sparkles className="w-4 h-4" />
                Mission
              </h3>
              <p className="text-sm">
                AI-powered exoplanet detection and classification using machine learning models
                trained on NASA's space telescope data from Kepler and TESS missions.
              </p>
            </div>
            <div>
              <h3 className="font-semibold text-white mb-2 flex items-center gap-2">
                <Database className="w-4 h-4" />
                Technology
              </h3>
              <p className="text-sm">
                Advanced ML algorithms including Random Forest, XGBoost, SVM, and Neural Networks
                for accurate exoplanet detection and classification with confidence scoring.
              </p>
            </div>
            <div>
              <h3 className="font-semibold text-white mb-2 flex items-center gap-2">
                <Telescope className="w-4 h-4" />
                Data Sources
              </h3>
              <ul className="text-sm space-y-1">
                <li>• NASA Exoplanet Archive</li>
                <li>• Kepler Space Telescope Data</li>
                <li>• TESS (Transiting Exoplanet Survey Satellite)</li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold text-white mb-2 flex items-center gap-2">
                <Globe className="w-4 h-4" />
                Features
              </h3>
              <ul className="text-sm space-y-1">
                <li>• Single target analysis</li>
                <li>• Batch CSV file processing</li>
                <li>• NASA Eyes 3D model integration</li>
                <li>• Confidence scoring & feature analysis</li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ExoplanetSearch;