"""
Machine Learning Models for Exoplanet Classification
Based on the ExoPlanetQuery functionality from the first repository
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import joblib
import os
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class ExoplanetMLModels:
    """Machine Learning models for exoplanet classification"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.trained = False
        
    def prepare_data(self, data: pd.DataFrame, 
                    handle_missing: str = "Fill with median", 
                    apply_scaling: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training by handling missing values and feature engineering
        """
        try:
            # Create a copy to avoid modifying original data
            df = data.copy()
            
            # Define feature columns (based on typical exoplanet data)
            feature_cols = []
            if 'pl_rade' in df.columns:
                feature_cols.append('pl_rade')  # Planet radius
            if 'pl_masse' in df.columns:
                feature_cols.append('pl_masse')  # Planet mass
            if 'pl_orbper' in df.columns:
                feature_cols.append('pl_orbper')  # Orbital period
            if 'pl_eqt' in df.columns:
                feature_cols.append('pl_eqt')  # Equilibrium temperature
            if 'st_rad' in df.columns:
                feature_cols.append('st_rad')  # Stellar radius
            if 'st_mass' in df.columns:
                feature_cols.append('st_mass')  # Stellar mass
            if 'pl_orbsmax' in df.columns:
                feature_cols.append('pl_orbsmax')  # Semi-major axis
            if 'pl_orbeccen' in df.columns:
                feature_cols.append('pl_orbeccen')  # Eccentricity
            if 'st_teff' in df.columns:
                feature_cols.append('st_teff')  # Stellar temperature
            
            # If no standard columns, try to identify numerical columns
            if not feature_cols:
                feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                # Remove any ID or target columns
                feature_cols = [col for col in feature_cols if not any(
                    keyword in col.lower() for keyword in ['id', 'target', 'class', 'type']
                )]
            
            self.feature_names = feature_cols
            
            # Create target variable if not present
            if 'exoplanet_type' not in df.columns:
                df['exoplanet_type'] = self._classify_exoplanet_type(df)
            
            # Handle missing values
            if handle_missing == "Drop rows":
                df = df.dropna(subset=feature_cols + ['exoplanet_type'])
            elif handle_missing == "Fill with median":
                for col in feature_cols:
                    if col in df.columns:
                        df[col] = df[col].fillna(df[col].median())
            elif handle_missing == "Fill with mean":
                for col in feature_cols:
                    if col in df.columns:
                        df[col] = df[col].fillna(df[col].mean())
            
            # Prepare features and target
            X = df[feature_cols].copy()
            y = df['exoplanet_type'].copy()
            
            # Apply scaling if requested
            if apply_scaling:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
                self.scalers['main'] = scaler
            
            logger.info(f"Data prepared: {len(X)} samples, {len(feature_cols)} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def _classify_exoplanet_type(self, df: pd.DataFrame) -> pd.Series:
        """
        Classify exoplanets based on their physical properties
        """
        exoplanet_types = []
        
        for _, row in df.iterrows():
            radius = row.get('pl_rade', 1.0)  # Default to Earth radius
            mass = row.get('pl_masse', 1.0)    # Default to Earth mass
            period = row.get('pl_orbper', 365.0)  # Default to Earth year
            temp = row.get('pl_eqt', 300.0)    # Default to ~Earth temperature
            
            # Classification logic based on physical properties
            if radius > 4.0:
                if mass > 20.0:
                    exoplanet_type = "Gas Giant"
                else:
                    exoplanet_type = "Neptune-like"
            elif radius > 1.5:
                if temp > 1000:
                    exoplanet_type = "Hot Super-Earth"
                else:
                    exoplanet_type = "Super-Earth"
            elif radius < 0.5:
                exoplanet_type = "Sub-Earth"
            else:
                if temp > 1000 and period < 10:
                    exoplanet_type = "Hot Jupiter"
                elif temp > 500:
                    exoplanet_type = "Warm Planet"
                else:
                    exoplanet_type = "Terrestrial"
            
            exoplanet_types.append(exoplanet_type)
        
        return pd.Series(exoplanet_types, index=df.index)
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                   model_type: str = "Random Forest",
                   test_size: float = 0.2,
                   random_state: int = 42) -> Optional[Dict[str, Any]]:
        """
        Train a specific model type
        """
        try:
            # Encode target labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
            )
            
            # Initialize model based on type
            if model_type == "Random Forest":
                model = RandomForestClassifier(
                    n_estimators=100, random_state=random_state, n_jobs=-1
                )
            elif model_type == "XGBoost":
                model = xgb.XGBClassifier(
                    n_estimators=100, random_state=random_state, n_jobs=-1
                )
            elif model_type == "SVM":
                model = SVC(
                    kernel='rbf', probability=True, random_state=random_state
                )
            elif model_type == "Logistic Regression":
                model = LogisticRegression(
                    max_iter=1000, random_state=random_state, n_jobs=-1
                )
            elif model_type == "Neural Network":
                model = MLPClassifier(
                    hidden_layer_sizes=(100, 50), max_iter=1000, 
                    random_state=random_state
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y_encoded, cv=5, scoring='accuracy')
            
            # Store model
            self.models[model_type] = model
            
            # Create results dictionary
            results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            # Add feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(X.columns, model.feature_importances_))
                results['feature_importance'] = feature_importance
            
            self.trained = True
            logger.info(f"Model {model_type} trained successfully. Accuracy: {accuracy:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training model {model_type}: {str(e)}")
            return None
    
    def predict_single(self, data: Dict[str, float], model_name: str = None) -> Dict[str, Any]:
        """
        Make prediction for a single exoplanet
        """
        try:
            if not self.trained or not self.models:
                raise ValueError("No trained models available")
            
            # Use best model if none specified
            if model_name is None:
                model_name = list(self.models.keys())[0]
            
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            model = self.models[model_name]
            
            # Prepare input data
            input_df = pd.DataFrame([data])
            
            # Apply scaling if available
            if 'main' in self.scalers:
                scaler = self.scalers['main']
                # Ensure all required features are present
                for feature in self.feature_names:
                    if feature not in input_df.columns:
                        input_df[feature] = 0.0  # Default value
                
                input_scaled = scaler.transform(input_df[self.feature_names])
                input_df = pd.DataFrame(input_scaled, columns=self.feature_names)
            
            # Make prediction
            prediction_encoded = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]
            
            # Decode prediction
            prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
            confidence = float(np.max(prediction_proba))
            
            result = {
                'prediction': prediction,
                'confidence': confidence,
                'model_used': model_name
            }
            
            # Add feature importance for this prediction if available
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_names, model.feature_importances_))
                result['feature_importance'] = feature_importance
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def predict_batch(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple exoplanets
        """
        try:
            results = []
            
            for _, row in data.iterrows():
                row_dict = row.to_dict()
                prediction = self.predict_single(row_dict)
                prediction['target'] = row.get('target', f'Unknown_{len(results)}')
                results.append(prediction)
            
            return results
            
        except Exception as e:
            logger.error(f"Error making batch predictions: {str(e)}")
            raise
    
    def get_model_performance(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance metrics for all trained models
        """
        if not hasattr(self, '_performance_data'):
            return {}
        return self._performance_data
    
    def save_models(self, filepath: str):
        """Save trained models to disk"""
        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names,
                'trained': self.trained
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Models saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise
    
    def load_models(self, filepath: str):
        """Load trained models from disk"""
        try:
            if os.path.exists(filepath):
                model_data = joblib.load(filepath)
                self.models = model_data['models']
                self.scalers = model_data['scalers']
                self.label_encoder = model_data['label_encoder']
                self.feature_names = model_data['feature_names']
                self.trained = model_data['trained']
                logger.info(f"Models loaded from {filepath}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False