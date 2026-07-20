import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import patch, MagicMock
import sys
import os
import PIL.Image

# ---------------------------------------------------------------------------
# Streamlit must be mocked BEFORE importing app, because app.py calls
# st.cache_resource at decoration time (module level). We make cache_resource
# a transparent pass-through so the real functions are preserved.
# ---------------------------------------------------------------------------
mock_st = MagicMock()
mock_st.file_uploader.return_value = None

def _passthrough_cache_resource(*args, **kwargs):
    """Transparent replacement for st.cache_resource — just returns the function unchanged."""
    def decorator(func):
        return func
    return decorator

mock_st.cache_resource = _passthrough_cache_resource
sys.modules['streamlit'] = mock_st
sys.modules['google.generativeai'] = MagicMock()

# Add parent directory to path to import app.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import app


def test_load_xception_model_architecture():
    """Test that the Xception model architecture builds correctly.

    We patch:
      - tf.keras.applications.Xception  -> returns a tiny stub model so we
        never download ImageNet weights in CI.
      - Sequential.load_weights         -> no-op so no .h5 file is needed.
    """

    # Build a minimal stub that satisfies the Sequential constructor
    stub_base = tf.keras.Sequential([
        tf.keras.layers.Conv2D(4, (3, 3), padding='same', input_shape=(299, 299, 3)), # type: ignore
        tf.keras.layers.GlobalAveragePooling2D(),
    ])
    stub_base.trainable = False

    with patch('app.tf.keras.applications.Xception', return_value=stub_base), \
         patch('app.Sequential.load_weights'):

        model = app.load_xception_model('dummy_path.h5')

    # Must be a real Sequential, not a Mock
    assert isinstance(model, tf.keras.models.Sequential), (
        f"Expected Sequential, got {type(model)}"
    )
    # Top layer must output 4 classes
    assert model.output_shape[-1] == 4


def test_generate_grad_cam_execution():
    """Test that the Grad-CAM generation runs without crashing and returns an image."""

    # Create a simple Sequential model with a Conv2D layer so Grad-CAM can find it
    mock_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(2, (3, 3), input_shape=(224, 224, 3), padding='same', name='conv2d_mock'), # type: ignore
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
