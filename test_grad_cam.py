import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D
from tensorflow.keras.optimizers import Adamax
import numpy as np
import cv2
import os

output_dir = 'saliency_maps'
os.makedirs(output_dir, exist_ok=True)

def load_xception_model(model_path):
  img_shape = (299, 299, 3)
  base_model = tf.keras.applications.Xception(include_top=False, weights="imagenet",
                                              input_shape=img_shape, pooling='max')

  model = Sequential([
    base_model,
    Flatten(),
    Dropout(rate=0.3),
    Dense(256, activation='relu'),
    Dropout(rate=0.25),
    Dense(4, activation='softmax')
  ])
  model.build((None,) + img_shape)
  return model

def get_target_layer(model):
    if isinstance(model.layers[0], tf.keras.Model):
        base_model = model.layers[0]
        for layer in reversed(base_model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D)):
                return layer
    else:
        for layer in reversed(model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D)):
                return layer
    return None

def generate_grad_cam(model, img_array, class_index, img_size, img_array_original, filename):
    is_nested = isinstance(model.layers[0], tf.keras.Model)
    target_layer = get_target_layer(model)
    
    with tf.GradientTape() as tape:
        if is_nested:
            base_model = model.layers[0]
            # Use base_model.layers[-1].output or base_model.outputs
            base_output_tensor = base_model.outputs[0] if hasattr(base_model, 'outputs') and base_model.outputs else base_model.layers[-1].output
            base_grad_model = tf.keras.Model(base_model.inputs, [target_layer.output, base_output_tensor])
            conv_outputs, base_outputs = base_grad_model(img_array)
            
            tape.watch(conv_outputs)
            
            x = base_outputs
            for layer in model.layers[1:]:
                x = layer(x)
            predictions = x
        else:
            final_output_tensor = model.outputs[0] if hasattr(model, 'outputs') and model.outputs else model.layers[-1].output
            grad_model = tf.keras.Model(model.inputs, [target_layer.output, final_output_tensor])
            conv_outputs, predictions = grad_model(img_array)
            
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    heatmap = cv2.resize(heatmap, img_size)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    superimposed_img = heatmap * 0.4 + img_array_original * 0.6
    superimposed_img = superimposed_img.astype(np.uint8)
    
    saliency_map_path = os.path.join(output_dir, filename)
    cv2.imwrite(saliency_map_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
    
    return superimposed_img

try:
    print("Testing CNN...")
    cnn_model = load_model('cnn_model.h5')
    dummy_img_cnn = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    dummy_array_cnn = np.expand_dims(dummy_img_cnn.astype(np.float32) / 255.0, axis=0)
    generate_grad_cam(cnn_model, dummy_array_cnn, 0, (224, 224), dummy_img_cnn, "cnn_cam.jpg")
    print("CNN Grad-CAM successful!")

    print("Testing Xception...")
    xc_model = load_xception_model('xception_model.weights.h5')
    dummy_img_xc = np.random.randint(0, 255, (299, 299, 3), dtype=np.uint8)
    dummy_array_xc = np.expand_dims(dummy_img_xc.astype(np.float32) / 255.0, axis=0)
    generate_grad_cam(xc_model, dummy_array_xc, 0, (299, 299), dummy_img_xc, "xc_cam.jpg")
    print("Xception Grad-CAM successful!")
    
except Exception as e:
    import traceback
    traceback.print_exc()
