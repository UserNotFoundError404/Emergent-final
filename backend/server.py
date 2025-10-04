from fastapi import FastAPI, APIRouter, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
from io import StringIO
import asyncio
import threading
import time

# Import our ML modules
from ml_models import ExoplanetMLModels
from data_loader import DataLoader

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI(title="ExoPlanet AI Classifier API", version="1.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Initialize ML models and data loader
ml_models = ExoplanetMLModels()
data_loader = DataLoader()

# Global variable to track training status
training_status = {
    "is_training": False,
    "progress": 0,
    "message": "Not started",
    "trained": False
}

# Define Models
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str

class ExoplanetPredictionRequest(BaseModel):
    pl_rade: Optional[float] = None     # Planet radius (Earth radii)
    pl_masse: Optional[float] = None    # Planet mass (Earth masses) 
    pl_orbper: Optional[float] = None   # Orbital period (days)
    pl_eqt: Optional[float] = None      # Equilibrium temperature (K)
    st_rad: Optional[float] = None      # Stellar radius (Solar radii)
    st_mass: Optional[float] = None     # Stellar mass (Solar masses)
    pl_orbsmax: Optional[float] = None  # Semi-major axis (AU)
    pl_orbeccen: Optional[float] = None # Eccentricity
    st_teff: Optional[float] = None     # Stellar temperature (K)

class TrainingRequest(BaseModel):
    data_sources: List[str] = ["Kepler Confirmed Planets", "TESS Objects of Interest"]
    load_limit: int = 2000
    models: List[str] = ["Random Forest", "XGBoost"]
    test_size: float = 0.2
    feature_scaling: bool = True
    handle_missing: str = "Fill with median"

class ExoplanetAnalysisResult(BaseModel):
    target: str
    classification: str
    confidence: float
    features: Optional[Dict[str, float]] = None
    model_used: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Basic routes
@api_router.get("/")
async def root():
    return {"message": "ExoPlanet AI Classifier API", "status": "running"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

# Training endpoints
@api_router.post("/train-models")
async def train_models(request: TrainingRequest):
    """Train ML models on NASA exoplanet data"""
    global training_status
    
    if training_status["is_training"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    def train_in_background():
        global training_status
        try:
            training_status["is_training"] = True
            training_status["progress"] = 0
            training_status["message"] = "Loading data from NASA archives..."
            
            # Load data from selected sources
            all_data = []
            
            for i, source in enumerate(request.data_sources):
                training_status["message"] = f"Loading {source}..."
                
                if source == "Kepler Confirmed Planets":
                    data = data_loader.load_kepler_confirmed_planets(limit=request.load_limit)
                elif source == "Kepler KOI Cumulative":
                    data = data_loader.load_kepler_koi_cumulative(limit=request.load_limit)
                elif source == "TESS Objects of Interest":
                    data = data_loader.load_tess_toi(limit=request.load_limit)
                elif source == "Planetary Systems":
                    data = data_loader.load_planetary_systems(limit=request.load_limit)
                else:
                    continue
                
                if data is not None and not data.empty:
                    data['data_source'] = source.replace(' ', '_')
                    all_data.append(data)
                
                training_status["progress"] = (i + 1) / len(request.data_sources) * 30
            
            if not all_data:
                training_status["message"] = "No data loaded"
                training_status["is_training"] = False
                return
            
            # Combine datasets
            training_status["message"] = "Combining datasets..."
            combined_data = data_loader.combine_datasets(all_data)
            
            # Prepare data for training
            training_status["message"] = "Preparing data for training..."
            training_status["progress"] = 35
            X, y = ml_models.prepare_data(
                combined_data,
                handle_missing=request.handle_missing,
                apply_scaling=request.feature_scaling
            )
            
            # Train models
            training_results = {}
            for i, model_name in enumerate(request.models):
                training_status["message"] = f"Training {model_name}..."
                training_status["progress"] = 35 + (i / len(request.models)) * 60
                
                result = ml_models.train_model(
                    X, y,
                    model_type=model_name,
                    test_size=request.test_size
                )
                
                if result:
                    training_results[model_name] = result
            
            training_status["progress"] = 100
            training_status["message"] = f"Training complete! {len(training_results)} models trained."
            training_status["trained"] = len(training_results) > 0
            training_status["is_training"] = False
            
            # Save models
            model_path = ROOT_DIR / "trained_models.pkl"
            ml_models.save_models(str(model_path))
            
        except Exception as e:
            training_status["is_training"] = False
            training_status["message"] = f"Training failed: {str(e)}"
            logging.error(f"Training error: {str(e)}")
    
    # Start training in background thread
    thread = threading.Thread(target=train_in_background)
    thread.daemon = True
    thread.start()
    
    return {"status": "Training started", "message": "Check /api/training-status for progress"}

@api_router.get("/training-status")
async def get_training_status():
    """Get current training status"""
    return training_status

# Load pre-trained models on startup
@app.on_event("startup")
async def load_models():
    """Load pre-trained models if available"""
    model_path = ROOT_DIR / "trained_models.pkl"
    if ml_models.load_models(str(model_path)):
        training_status["trained"] = True
        training_status["message"] = "Pre-trained models loaded"
        logging.info("Pre-trained models loaded successfully")
    else:
        # Train with sample data for immediate functionality
        try:
            logging.info("No pre-trained models found. Training with sample data...")
            sample_data = data_loader._generate_sample_data(500, 'Sample')
            X, y = ml_models.prepare_data(sample_data)
            
            # Train a simple model
            result = ml_models.train_model(X, y, model_type="Random Forest")
            if result:
                training_status["trained"] = True
                training_status["message"] = "Sample model trained and ready"
                logging.info("Sample model trained successfully")
        except Exception as e:
            logging.error(f"Error training sample model: {str(e)}")

# Analysis endpoints
@api_router.post("/exoplanet-analysis")
async def analyze_exoplanet(
    target_name: Optional[str] = Form(None),
    analysis_type: str = Form("single_target"),
    file: Optional[UploadFile] = File(None)
):
    """Analyze exoplanet(s) using trained ML models"""
    
    if not training_status["trained"]:
        raise HTTPException(status_code=400, detail="No trained models available. Please train models first.")
    
    try:
        if analysis_type == "csv_batch" and file:
            # Batch analysis from CSV file
            content = await file.read()
            csv_data = StringIO(content.decode('utf-8'))
            df = pd.read_csv(csv_data)
            
            results = ml_models.predict_batch(df)
            
            # Store results in database
            analysis_record = {
                "analysis_type": "batch",
                "file_name": file.filename,
                "num_targets": len(results),
                "timestamp": datetime.utcnow(),
                "results": results
            }
            await db.analyses.insert_one(analysis_record)
            
            return {"batch_results": results, "analysis_type": "batch"}
        
        elif analysis_type == "single_target" and target_name:
            # Single target analysis
            
            # First try to get real data from NASA archive
            planet_data = data_loader.search_planet_by_name(target_name)
            
            if not planet_data:
                # If no real data found, create a basic entry
                planet_data = {
                    'pl_name': target_name,
                    'pl_rade': 1.0,    # Default values for prediction
                    'pl_masse': 1.0,
                    'pl_orbper': 365.0,
                    'pl_eqt': 300.0,
                    'st_rad': 1.0,
                    'st_mass': 1.0,
                    'pl_orbsmax': 1.0,
                    'pl_orbeccen': 0.0,
                    'st_teff': 5778.0
                }
            
            # Make prediction
            prediction = ml_models.predict_single(planet_data)
            
            # Store analysis in database
            analysis_record = {
                "analysis_type": "single",
                "target_name": target_name,
                "timestamp": datetime.utcnow(),
                "prediction": prediction,
                "planet_data": planet_data
            }
            await db.analyses.insert_one(analysis_record)
            
            result = {
                "target": target_name,
                "classification": prediction["prediction"],
                "confidence": prediction["confidence"],
                "features": planet_data,
                "model_used": prediction["model_used"],
                "timestamp": datetime.utcnow()
            }
            
            return result
        
        else:
            raise HTTPException(status_code=400, detail="Invalid analysis request. Provide either target_name or CSV file.")
    
    except Exception as e:
        logging.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@api_router.post("/predict-exoplanet")
async def predict_exoplanet(request: ExoplanetPredictionRequest):
    """Make prediction based on provided exoplanet parameters"""
    
    if not training_status["trained"]:
        raise HTTPException(status_code=400, detail="No trained models available. Please train models first.")
    
    try:
        # Convert request to dictionary
        data = request.dict()
        
        # Remove None values and use defaults
        clean_data = {}
        for key, value in data.items():
            if value is not None:
                clean_data[key] = value
        
        # Make prediction
        prediction = ml_models.predict_single(clean_data)
        
        result = ExoplanetAnalysisResult(
            target="Custom Input",
            classification=prediction["prediction"],
            confidence=prediction["confidence"],
            features=clean_data,
            model_used=prediction["model_used"]
        )
        
        return result
        
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@api_router.get("/analysis-history")
async def get_analysis_history(limit: int = 10):
    """Get recent analysis history"""
    try:
        analyses = await db.analyses.find().sort("timestamp", -1).limit(limit).to_list(limit)
        return {"analyses": analyses}
    except Exception as e:
        logging.error(f"Error retrieving analysis history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analysis history")

@api_router.get("/model-performance")
async def get_model_performance():
    """Get performance metrics of trained models"""
    if not training_status["trained"]:
        raise HTTPException(status_code=400, detail="No trained models available")
    
    try:
        performance = ml_models.get_model_performance()
        return {"performance": performance, "models_available": list(ml_models.models.keys())}
    except Exception as e:
        logging.error(f"Error getting model performance: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get model performance")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
