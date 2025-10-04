import React, { useState, useEffect } from "react";
import "@/App.css";
import { Telescope, Sparkles, Database, Github, Home } from "lucide-react";
import ExoplanetSearch from "@/components/ExoplanetSearch";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [showHomepage, setShowHomepage] = useState(true);

  // Test backend connection
  useEffect(() => {
    const testConnection = async () => {
      try {
        const response = await axios.get(`${API}/`);
        console.log('Backend connected:', response.data.message);
      } catch (e) {
        console.error('Backend connection error:', e);
      }
    };
    testConnection();
  }, []);

  if (showHomepage) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-950">
        {/* Header */}
        <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 py-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Telescope className="w-8 h-8 text-blue-400" />
                <div>
                  <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 via-cyan-400 to-blue-400 bg-clip-text text-transparent">
                    ExoPlanet AI Classifier
                  </h1>
                  <p className="text-slate-400 text-sm">Advanced ML Exoplanet Detection & Analysis</p>
                </div>
              </div>
              <a
                href="https://github.com/UserNotFoundError404/REPLIT/tree/main/ExoPlanetQuery"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-lg transition-colors"
              >
                <Github className="w-5 h-5" />
                <span className="hidden sm:inline">View Code</span>
              </a>
            </div>
          </div>
        </header>

        {/* Hero Section */}
        <main className="max-w-7xl mx-auto px-4">
          <div className="min-h-[calc(100vh-200px)] flex flex-col items-center justify-center text-center py-20">
            <div className="mb-8 relative">
              <div className="absolute inset-0 blur-3xl opacity-30 bg-gradient-to-r from-blue-500 via-cyan-500 to-blue-500 rounded-full"></div>
              <Telescope className="w-24 h-24 text-cyan-400 relative z-10" />
            </div>

            <h1 className="text-5xl md:text-7xl font-bold mb-6 bg-gradient-to-r from-blue-400 via-cyan-300 to-blue-400 bg-clip-text text-transparent leading-tight">
              Discover New Worlds
            </h1>

            <p className="text-xl md:text-2xl text-slate-300 mb-4 max-w-3xl">
              AI-powered exoplanet detection using NASA's space telescope data
            </p>

            <p className="text-lg text-slate-400 mb-12 max-w-2xl">
              Train machine learning models on real NASA Kepler and TESS datasets to classify exoplanets 
              into categories like Hot Jupiter, Super-Earth, Neptune-like, and explore them in 3D.
            </p>

            <button
              onClick={() => setShowHomepage(false)}
              className="group px-8 py-4 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white text-lg font-semibold rounded-lg transition-all shadow-lg hover:shadow-cyan-500/50 hover:scale-105"
            >
              <span className="flex items-center gap-3">
                Start Exploring
                <Sparkles className="w-5 h-5 group-hover:rotate-12 transition-transform" />
              </span>
            </button>
          </div>

          {/* Features Grid */}
          <div className="grid md:grid-cols-3 gap-8 pb-20">
            <div className="bg-slate-900/70 backdrop-blur-sm rounded-xl p-8 border border-slate-800 hover:border-blue-500/50 transition-all">
              <div className="w-12 h-12 bg-blue-500/10 rounded-lg flex items-center justify-center mb-4">
                <Sparkles className="w-6 h-6 text-blue-400" />
              </div>
              <h3 className="text-xl font-bold text-white mb-3">AI Detection</h3>
              <p className="text-slate-400">
                Multiple ML models including Random Forest, XGBoost, SVM, and Neural Networks trained on NASA's Kepler and TESS mission data.
              </p>
            </div>

            <div className="bg-slate-900/70 backdrop-blur-sm rounded-xl p-8 border border-slate-800 hover:border-cyan-500/50 transition-all">
              <div className="w-12 h-12 bg-cyan-500/10 rounded-lg flex items-center justify-center mb-4">
                <Database className="w-6 h-6 text-cyan-400" />
              </div>
              <h3 className="text-xl font-bold text-white mb-3">Real NASA Data</h3>
              <p className="text-slate-400">
                Downloads and processes data directly from NASA's Exoplanet Archive, Kepler confirmed planets, and TESS objects of interest.
              </p>
            </div>

            <div className="bg-slate-900/70 backdrop-blur-sm rounded-xl p-8 border border-slate-800 hover:border-blue-500/50 transition-all">
              <div className="w-12 h-12 bg-blue-500/10 rounded-lg flex items-center justify-center mb-4">
                <Telescope className="w-6 h-6 text-blue-400" />
              </div>
              <h3 className="text-xl font-bold text-white mb-3">3D Visualization</h3>
              <p className="text-slate-400">
                Integrated with NASA Eyes on Exoplanets for interactive 3D models and visualizations of discovered worlds.
              </p>
            </div>
          </div>
        </main>

        {/* Footer */}
        <footer className="border-t border-slate-800 bg-slate-900/50 backdrop-blur-sm">
          <div className="max-w-7xl mx-auto px-4 py-6 text-center text-slate-400 text-sm">
            <p>
              Built with NASA's open data for exoplanet research •{' '}
              <a href="https://exoplanetarchive.ipac.caltech.edu/" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:text-blue-300">
                NASA Exoplanet Archive
              </a>
            </p>
          </div>
        </footer>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-950">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <button
              onClick={() => setShowHomepage(true)}
              className="flex items-center gap-3 hover:opacity-80 transition-opacity"
            >
              <Telescope className="w-8 h-8 text-blue-400" />
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 via-cyan-400 to-blue-400 bg-clip-text text-transparent">
                  ExoPlanet AI Classifier
                </h1>
                <p className="text-slate-400 text-sm">Advanced ML Exoplanet Detection & Analysis</p>
              </div>
            </button>
            <div className="flex items-center gap-3">
              <button
                onClick={() => setShowHomepage(true)}
                className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-lg transition-colors"
              >
                <Home className="w-5 h-5" />
                <span className="hidden sm:inline">Home</span>
              </button>
              <a
                href="https://github.com/UserNotFoundError404/REPLIT/tree/main/ExoPlanetQuery"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-lg transition-colors"
              >
                <Github className="w-5 h-5" />
                <span className="hidden sm:inline">View Code</span>
              </a>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        <ExoplanetSearch />
      </main>

      {/* Footer */}
      <footer className="mt-16 border-t border-slate-800 bg-slate-900/50 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 py-6 text-center text-slate-400 text-sm">
          <p>
            Built with NASA's open data for exoplanet research •{' '}
            <a href="https://exoplanetarchive.ipac.caltech.edu/" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:text-blue-300">
              NASA Exoplanet Archive
            </a>
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
