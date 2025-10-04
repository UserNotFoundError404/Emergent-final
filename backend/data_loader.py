"""
Data Loader for NASA Exoplanet Archive Data
Based on the ExoPlanetQuery functionality
"""

import pandas as pd
import numpy as np
import requests
from io import StringIO
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlencode
import time

logger = logging.getLogger(__name__)

class DataLoader:
    """Load data from NASA Exoplanet Archive and other sources"""
    
    def __init__(self):
        self.base_urls = {
            'nasa_archive': 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI',
            'confirmed_planets': 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI',
            'koi_cumulative': 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI',
            'tess_toi': 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI',
        }
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'ExoplanetMLClassifier/1.0'})
    
    def load_kepler_confirmed_planets(self, limit: int = 2000) -> Optional[pd.DataFrame]:
        """
        Load confirmed Kepler planets from NASA Exoplanet Archive
        """
        try:
            params = {
                'table': 'pscomppars',
                'select': 'pl_name,pl_rade,pl_masse,pl_orbper,pl_eqt,st_rad,st_mass,pl_orbsmax,pl_orbeccen,st_teff,sy_dist,disc_facility',
                'where': 'disc_facility like "%Kepler%" and pl_rade is not null',
                'format': 'csv',
                'order': 'pl_name'
            }
            
            url = f"{self.base_urls['confirmed_planets']}?{urlencode(params)}"
            
            logger.info("Loading Kepler confirmed planets...")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse CSV data
            df = pd.read_csv(StringIO(response.text), comment='#')
            
            if len(df) == 0:
                logger.warning("No Kepler confirmed planets data retrieved")
                return None
            
            # Apply limit
            if limit and len(df) > limit:
                df = df.head(limit)
            
            # Clean and process data
            df = self._clean_planet_data(df)
            
            logger.info(f"Successfully loaded {len(df)} Kepler confirmed planets")
            return df
            
        except Exception as e:
            logger.error(f"Error loading Kepler confirmed planets: {str(e)}")
            return self._generate_sample_data(limit, 'Kepler')
    
    def load_kepler_koi_cumulative(self, limit: int = 2000) -> Optional[pd.DataFrame]:
        """
        Load Kepler Objects of Interest (KOI) cumulative table
        """
        try:
            params = {
                'table': 'cumulative',
                'select': 'kepoi_name,koi_prad,koi_period,koi_teq,koi_sma,koi_eccen,koi_disposition',
                'where': 'koi_prad is not null',
                'format': 'csv',
                'order': 'kepoi_name'
            }
            
            url = f"{self.base_urls['koi_cumulative']}?{urlencode(params)}"
            
            logger.info("Loading Kepler KOI cumulative data...")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            df = pd.read_csv(StringIO(response.text), comment='#')
            
            if len(df) == 0:
                logger.warning("No KOI cumulative data retrieved")
                return None
            
            # Apply limit
            if limit and len(df) > limit:
                df = df.head(limit)
            
            # Rename columns to match standard format
            column_mapping = {
                'kepoi_name': 'pl_name',
                'koi_prad': 'pl_rade',
                'koi_period': 'pl_orbper',
                'koi_teq': 'pl_eqt',
                'koi_sma': 'pl_orbsmax',
                'koi_eccen': 'pl_orbeccen'
            }
            df = df.rename(columns=column_mapping)
            
            # Clean and process data
            df = self._clean_planet_data(df)
            
            logger.info(f"Successfully loaded {len(df)} KOI objects")
            return df
            
        except Exception as e:
            logger.error(f"Error loading KOI cumulative data: {str(e)}")
            return self._generate_sample_data(limit, 'KOI')
    
    def load_tess_toi(self, limit: int = 2000) -> Optional[pd.DataFrame]:
        """
        Load TESS Objects of Interest (TOI)
        """
        try:
            params = {
                'table': 'toi',
                'select': 'toi,pl_rade,pl_orbper,pl_eqt,pl_orbsmax,pl_orbeccen,st_rad,st_mass,st_teff',
                'where': 'pl_rade is not null',
                'format': 'csv',
                'order': 'toi'
            }
            
            url = f"{self.base_urls['tess_toi']}?{urlencode(params)}"
            
            logger.info("Loading TESS TOI data...")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            df = pd.read_csv(StringIO(response.text), comment='#')
            
            if len(df) == 0:
                logger.warning("No TESS TOI data retrieved")
                return None
            
            # Apply limit
            if limit and len(df) > limit:
                df = df.head(limit)
            
            # Rename columns to match standard format
            column_mapping = {
                'toi': 'pl_name'
            }
            df = df.rename(columns=column_mapping)
            
            # Clean and process data
            df = self._clean_planet_data(df)
            
            logger.info(f"Successfully loaded {len(df)} TESS TOI objects")
            return df
            
        except Exception as e:
            logger.error(f"Error loading TESS TOI data: {str(e)}")
            return self._generate_sample_data(limit, 'TESS')
    
    def load_planetary_systems(self, limit: int = 2000) -> Optional[pd.DataFrame]:
        """
        Load planetary systems composite parameters
        """
        try:
            params = {
                'table': 'ps',
                'select': 'pl_name,pl_rade,pl_masse,pl_orbper,pl_eqt,st_rad,st_mass,pl_orbsmax,pl_orbeccen,st_teff',
                'where': 'pl_rade is not null and pl_masse is not null',
                'format': 'csv',
                'order': 'pl_name'
            }
            
            url = f"{self.base_urls['nasa_archive']}?{urlencode(params)}"
            
            logger.info("Loading planetary systems data...")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            df = pd.read_csv(StringIO(response.text), comment='#')
            
            if len(df) == 0:
                logger.warning("No planetary systems data retrieved")
                return None
            
            # Apply limit
            if limit and len(df) > limit:
                df = df.head(limit)
            
            # Clean and process data
            df = self._clean_planet_data(df)
            
            logger.info(f"Successfully loaded {len(df)} planetary systems")
            return df
            
        except Exception as e:
            logger.error(f"Error loading planetary systems data: {str(e)}")
            return self._generate_sample_data(limit, 'Systems')
    
    def combine_datasets(self, datasets: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Combine multiple datasets into one
        """
        try:
            if not datasets:
                return pd.DataFrame()
            
            combined_df = pd.concat(datasets, ignore_index=True)
            
            # Remove duplicates based on planet name
            if 'pl_name' in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=['pl_name'], keep='first')
            
            # Remove rows with all NaN values in key columns
            key_columns = ['pl_rade', 'pl_masse', 'pl_orbper']
            available_columns = [col for col in key_columns if col in combined_df.columns]
            
            if available_columns:
                combined_df = combined_df.dropna(subset=available_columns, how='all')
            
            logger.info(f"Combined datasets: {len(combined_df)} total records")
            return combined_df
            
        except Exception as e:
            logger.error(f"Error combining datasets: {str(e)}")
            return pd.DataFrame()
    
    def _clean_planet_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize planet data
        """
        try:
            # Convert numeric columns
            numeric_columns = ['pl_rade', 'pl_masse', 'pl_orbper', 'pl_eqt', 
                              'st_rad', 'st_mass', 'pl_orbsmax', 'pl_orbeccen', 'st_teff']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove extreme outliers
            if 'pl_rade' in df.columns:
                df = df[(df['pl_rade'] > 0) & (df['pl_rade'] < 50)]  # Reasonable radius range
            
            if 'pl_masse' in df.columns:
                df = df[(df['pl_masse'] > 0) & (df['pl_masse'] < 5000)]  # Reasonable mass range
            
            if 'pl_orbper' in df.columns:
                df = df[(df['pl_orbper'] > 0) & (df['pl_orbper'] < 50000)]  # Reasonable period range
            
            # Fill missing planet names
            if 'pl_name' not in df.columns or df['pl_name'].isna().all():
                df['pl_name'] = [f'Planet_{i:04d}' for i in range(len(df))]
            else:
                df['pl_name'] = df['pl_name'].fillna([f'Planet_{i:04d}' for i in range(len(df))])
            
            return df.reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Error cleaning planet data: {str(e)}")
            return df
    
    def _generate_sample_data(self, n_samples: int = 100, source: str = 'Sample') -> pd.DataFrame:
        """
        Generate sample exoplanet data for testing/fallback
        """
        try:
            np.random.seed(42)  # For reproducible results
            
            data = {
                'pl_name': [f'{source}_Planet_{i:04d}' for i in range(n_samples)],
                'pl_rade': np.random.lognormal(0, 0.5, n_samples),  # Planet radius (Earth radii)
                'pl_masse': np.random.lognormal(0, 1.0, n_samples),  # Planet mass (Earth masses)
                'pl_orbper': np.random.lognormal(2, 2.0, n_samples),  # Orbital period (days)
                'pl_eqt': np.random.normal(800, 400, n_samples),     # Equilibrium temperature (K)
                'st_rad': np.random.normal(1.0, 0.3, n_samples),    # Stellar radius (Solar radii)
                'st_mass': np.random.normal(1.0, 0.2, n_samples),   # Stellar mass (Solar masses)
                'pl_orbsmax': np.random.lognormal(-1, 1.0, n_samples),  # Semi-major axis (AU)
                'pl_orbeccen': np.random.beta(2, 8, n_samples),     # Eccentricity
                'st_teff': np.random.normal(5500, 800, n_samples),  # Stellar temperature (K)
            }
            
            df = pd.DataFrame(data)
            
            # Ensure positive values where needed
            df['pl_rade'] = np.abs(df['pl_rade'])
            df['pl_masse'] = np.abs(df['pl_masse'])
            df['pl_orbper'] = np.abs(df['pl_orbper'])
            df['st_rad'] = np.abs(df['st_rad'])
            df['st_mass'] = np.abs(df['st_mass'])
            df['pl_orbsmax'] = np.abs(df['pl_orbsmax'])
            df['st_teff'] = np.abs(df['st_teff'])
            
            logger.info(f"Generated {n_samples} sample exoplanet records for {source}")
            return df
            
        except Exception as e:
            logger.error(f"Error generating sample data: {str(e)}")
            return pd.DataFrame()
    
    def search_planet_by_name(self, planet_name: str) -> Optional[Dict[str, Any]]:
        """
        Search for a specific planet by name in NASA archive
        """
        try:
            params = {
                'table': 'pscomppars',
                'select': 'pl_name,pl_rade,pl_masse,pl_orbper,pl_eqt,st_rad,st_mass,pl_orbsmax,pl_orbeccen,st_teff',
                'where': f'pl_name like "%{planet_name}%"',
                'format': 'csv'
            }
            
            url = f"{self.base_urls['nasa_archive']}?{urlencode(params)}"
            
            logger.info(f"Searching for planet: {planet_name}")
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            df = pd.read_csv(StringIO(response.text), comment='#')
            
            if len(df) == 0:
                logger.warning(f"No data found for planet: {planet_name}")
                return None
            
            # Return first match as dictionary
            planet_data = df.iloc[0].to_dict()
            
            # Convert any NaN values to None
            for key, value in planet_data.items():
                if pd.isna(value):
                    planet_data[key] = None
            
            logger.info(f"Found planet data for: {planet_name}")
            return planet_data
            
        except Exception as e:
            logger.error(f"Error searching for planet {planet_name}: {str(e)}")
            return None