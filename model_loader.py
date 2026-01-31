import os
import requests
import streamlit as st
from pathlib import Path

def download_file_from_url(url: str, destination: str) -> bool:
    """
    Download a file from a URL to a destination path.
    
    Args:
        url: URL to download from
        destination: Local path to save the file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Create directory if it doesn't exist
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
        
        return True
    except Exception as e:
        st.error(f"Error downloading file: {str(e)}")
        return False

def download_from_google_drive(file_id: str, destination: str) -> bool:
    """
    Download a file from Google Drive using the file ID.
    
    Args:
        file_id: Google Drive file ID
        destination: Local path to save the file
        
    Returns:
        bool: True if successful, False otherwise
    """
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)
    return True

def ensure_models_exist():
    """
    Ensure model files exist. Download from cloud storage if needed.
    Configure this with your actual model storage URLs.
    """
    model_path = "model.pkl"
    preprocessor_path = "preprocessor.pkl"
    
    # Check if models already exist
    if os.path.exists(model_path) and os.path.exists(preprocessor_path):
        return True
    
    # If using Streamlit secrets for URLs
    if hasattr(st, 'secrets'):
        try:
            # Option 1: Direct URL download
            if 'MODEL_URL' in st.secrets and not os.path.exists(model_path):
                with st.spinner("Downloading model file..."):
                    download_file_from_url(st.secrets['MODEL_URL'], model_path)
            
            if 'PREPROCESSOR_URL' in st.secrets and not os.path.exists(preprocessor_path):
                with st.spinner("Downloading preprocessor file..."):
                    download_file_from_url(st.secrets['PREPROCESSOR_URL'], preprocessor_path)
            
            # Option 2: Google Drive download
            if 'MODEL_GDRIVE_ID' in st.secrets and not os.path.exists(model_path):
                with st.spinner("Downloading model from Google Drive..."):
                    download_from_google_drive(st.secrets['MODEL_GDRIVE_ID'], model_path)
            
            if 'PREPROCESSOR_GDRIVE_ID' in st.secrets and not os.path.exists(preprocessor_path):
                with st.spinner("Downloading preprocessor from Google Drive..."):
                    download_from_google_drive(st.secrets['PREPROCESSOR_GDRIVE_ID'], preprocessor_path)
                    
        except Exception as e:
            st.error(f"Error loading models from cloud storage: {str(e)}")
            return False
    
    # Verify both files exist
    if os.path.exists(model_path) and os.path.exists(preprocessor_path):
        return True
    else:
        st.error("""
        ⚠️ Model files not found!
        
        Please ensure model.pkl and preprocessor.pkl are available, or configure 
        cloud storage URLs in Streamlit secrets.
        """)
        return False
