import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import patch
import sys
import os
import PIL.Image
from unittest.mock import MagicMock

# Mock streamlit and generativeai before importing app.py to prevent import errors in CI
mock_st = MagicMock()
mock_st.file_uploader.return_value = None
sys.modules['streamlit'] = mock_st
sys.modules['google.generativeai'] = MagicMock()

# Add parent directory to path to import app.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import app

def test_load_xception_model_architecture():
    """Test that the Xception model architecture builds correctly."""
    # We mock load_weights to avoid loading the 250MB weights file during CI
    with patch('app.Sequential.load_weights') as mock_load_weights:
        model = app.load_xception_model('dummy_path.h5')
        
        # Verify it's a Sequential model
        assert isinstance(model, tf.keras.models.Sequential)
        
        # Verify output shape is for 4 classes
        assert model.output_shape == (None, 4)
        
        # Verify load_weights was called with the correct path
        mock_load_weights.assert_called_once_with('dummy_path.h5')

def test_generate_grad_cam_execution():
    """Test that the Grad-CAM generation runs without crashing and returns an image."""
    
    # Create a simple Sequential model with a Conv2D layer so Grad-CAM can find it
    mock_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(2, (3,3), input_shape=(224, 224, 3), padding='same', name='conv2d_mock'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    
    # Dummy input image array (1, 224, 224, 3)
    img_array = np.random.rand(1, 224, 224, 3).astype(np.float32)
    
    # Dummy PIL Image
    img = PIL.Image.new('RGB', (224, 224), color='gray')
    
    # Dummy uploaded_file object to mock Streamlit's UploadedFile
    class DummyUploadedFile:
        def __init__(self):
            self.name = "test_image.jpg"
        def getbuffer(self):
            return b"dummy_bytes_for_testing"
            
    uploaded_file = DummyUploadedFile()
    
    # Generate the Grad-CAM heatmap
    saliency_map = app.generate_grad_cam(
        model=mock_model,
        img_array=img_array,
        class_index=1,
        img_size=(224, 224),
        img=img,
        uploaded_file=uploaded_file
    )
    
    # Verify the output is a valid image array of the correct shape
    assert isinstance(saliency_map, np.ndarray)
    assert saliency_map.shape == (224, 224, 3)
