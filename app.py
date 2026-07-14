import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.preprocessing import image # pyright: ignore[reportMissingImports]
import numpy as np
import plotly.graph_objects as go
import cv2
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Flatten # type: ignore
from tensorflow.keras.optimizers import Adamax # type: ignore
from tensorflow.keras.metrics import Precision, Recall # type: ignore
import google.generativeai as genai
import PIL.Image
import os
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) # pyright: ignore[reportPrivateImportUsage]

output_dir = 'saliency_maps'
os.makedirs(output_dir, exist_ok=True)





def generate_saliency_map(model, img_array, class_index, img_size, img, uploaded_file):
  with tf.GradientTape() as tape:
    img_tensor = tf.convert_to_tensor(img_array)
    tape.watch(img_tensor)
    predictions = model(img_tensor)
    target_class = predictions[:, class_index]

  gradients = tape.gradient(target_class, img_tensor)
  gradients = tf.math.abs(gradients) # type: ignore
  gradients = tf.reduce_max(gradients, axis=-1)
  gradients = gradients.numpy().squeeze()

  # Resize gradients to match original image size
  gradients = cv2.resize(gradients, img_size)

  # Create a circular mask for the brain area
  center = (gradients.shape[0] // 2, gradients.shape[1] // 2)
  radius = min(center[0], center[1]) - 10
  y, x = np.ogrid[:gradients.shape[0], :gradients.shape[1]]
  mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2

  # Apply mask to gradients
  gradients = gradients * mask

  # Normalize only the brain areal
  brain_gradients = gradients[mask]

  if brain_gradients.max() > brain_gradients.min():
    brain_gradients = (brain_gradients - brain_gradients.min()) / (brain_gradients.max() - brain_gradients.min())
  gradients[mask] = brain_gradients

  # Apply a higher threshold
  threshold = np.percentile(gradients[mask], 80)
  gradients[gradients < threshold] = 0

  # Apply more aggressive smoothing
  gradients = cv2.GaussianBlur(gradients, (11, 11), 0)

  # Create a heatmap overlay with enhanced contrast
  heatmap = cv2.applyColorMap(np.uint8(255 * gradients), cv2.COLORMAP_JET) # type: ignore
  heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

  # Resize heatmap to match original image size
  heatmap = cv2.resize(heatmap, img_size)

  # Superimpose the heatmap on original image with increased opacity
  original_img = image.img_to_array(img)
  superimposed_img = heatmap * 0.7 + original_img * 0.3
  superimposed_img = superimposed_img.astype(np.uint8)

  img_path = os.path.join(output_dir, uploaded_file.name)

  with open(img_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

  saliency_map_path = f'saliency_maps/{uploaded_file.name}'

  # Save the saliency map
  cv2.imwrite(saliency_map_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

  return superimposed_img


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

  model.compile(Adamax(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy',
                         Precision(),
                         Recall()]
                )

  model.load_weights(model_path)

  return model


st.title("Brain Tumor Classification")

st.write("Upload an image of a brain MRI scan to classify")

uploaded_file = st.file_uploader("Choose an Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

  selected_model = st.radio(
      "Selected Model",
      ("Transfer Learning - Xception", "Custom CNN", "Ensemble (Xception + CNN)")
  )

  if selected_model == "Transfer Learning - Xception":
    model = load_xception_model('xception_model.weights.h5')
    img_size = (299, 299)
  elif selected_model == "Custom CNN":
    model = load_model('cnn_model.h5')
    img_size = (224, 224)
  else:
    model_xc = load_xception_model('xception_model.weights.h5')
    model_cnn = load_model('cnn_model.h5')
    img_size = (299, 299)

  labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

  if selected_model == "Ensemble (Xception + CNN)":
    img_xc = image.load_img(uploaded_file, target_size=(299, 299))
    img_array_xc = image.img_to_array(img_xc)
    img_array_xc = np.expand_dims(img_array_xc, axis=0) / 255.0
    pred_xc = model_xc.predict(img_array_xc)

    img_cnn = image.load_img(uploaded_file, target_size=(224, 224))
    img_array_cnn = image.img_to_array(img_cnn)
    img_array_cnn = np.expand_dims(img_array_cnn, axis=0) / 255.0
    pred_cnn = model_cnn.predict(img_array_cnn)

    prediction = (pred_xc + pred_cnn) / 2
    
    # Use Xception for the saliency map in Ensemble mode
    model = model_xc
    img_size = (299, 299)
    img_array = img_array_xc
    img = img_xc
  else:
    img = image.load_img(uploaded_file, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)

  class_index = np.argmax(prediction[0])
  result = labels[class_index]

  st.write(f"Predicted Class: {result}")
  st.write("Predictions:")
  for label, prob in zip(labels, prediction[0]):
    st.write(f"{label}: {prob:.4f}")

  saliency_map = generate_saliency_map(model, img_array, class_index, img_size, img, uploaded_file)

  col1, col2 = st.columns(2)

  with col1:
    st.markdown(
        """
        <style>
        .img-container {
            border: 3px solid #FFFFFF;
            padding: 10px;
            border-radius: 8px;
        }
        </style>
        """, unsafe_allow_html=True)
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

  with col2:
    st.markdown(
        """
        <style>
        .img-container {
            border: 3px solid #FFFFFF;
            padding: 10px;
            border-radius: 8px;
        }
        </style>
        """, unsafe_allow_html=True)
    st.image(saliency_map, caption="Saliency Map", use_container_width=True)

  st.write("## Classification Results")

  result_container = st.container()
  result_container.markdown(
      f"""
      <div style="background-color: #000000; color: #ffffff; padding: 30px; border-radius: 15px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <div style="flex: 1; text-align: center;">
            <h3 style="color: #ffffff; margin-bottom: 10px; font-size: 20px;">Prediction</h3>
            <p style="font-size: 36px; font-weight: 800; color: #FF0000; margin: 0;">
              {result}
            </p>
          </div>
          <div style="width: 2px; height: 80px; background-color: #ffffff; margin: 0 20px;"></div>
          <div style="flex: 1; text-align: center;">
            <h3 style="color: #ffffff; margin-bottom: 10px; font-size: 20px;">Confidence</h3>
            <p style="font-size: 36px; font-weight: 800; color: #2196F3; margin: 0;">
              {prediction[0][class_index]:.4%}
            </p>
          </div>
        </div>
      </div>
      """,
      unsafe_allow_html=True)

  probabilities = prediction[0]
  sorted_indices = np.argsort(probabilities)[::-1]
  sorted_labels = [labels[i] for i in sorted_indices]
  sorted_probabilities = probabilities[sorted_indices]

  fig = go.Figure(go.Bar(
      x=sorted_probabilities,
      y=sorted_labels,
      orientation='h',
      marker_color=['red' if label == result else 'blue' for label in sorted_labels])
  )

  fig.update_layout(
      title='Probabilities for each class',
      xaxis_title='Probability',
      yaxis_title='Class',
      height=400,
      width=600,
      yaxis=dict(autorange="reversed")
  )

  for i, prob in enumerate(sorted_probabilities):
    fig.add_annotation(
        x=prob,
        y=i,
        text=f'{prob:.4f}',
        showarrow=False,
        xanchor='left',
        xshift=5
    )

  st.plotly_chart(fig)

  saliency_map_path = f'saliency_maps/{uploaded_file.name}'

  st.write("## Interactive Explanation")

  # Initialize chat history if file changed
  if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
      st.session_state.current_file = uploaded_file.name
      st.session_state.messages = []

  gemini_model = genai.GenerativeModel(model_name="gemini-3.5-flash") # type: ignore

  # Generate initial explanation if empty
  if len(st.session_state.messages) == 0:
      img = PIL.Image.open(saliency_map_path)
      prompt = f"""You are an expert neurologist. You are tasked with explaining a saliency map of a brain tumor MRI scan. The saliency map was generated by a deep learning model that was trained to classify brain tumors as either glioma, meningioma, pituitary, or no tumor.

The saliency map highlights the regions of the image that the machine learning model is focusing on to make the prediction.

The deep learning model predicted the image to be of class '{result}' with a confidence of {prediction[0][class_index] * 100}%.

In your response:
- Explain what regions of the brain the model is focusing on, based on the saliency map. Refer to the regions highlighted in light cyan.
- Explain possible reasons why the model made the prediction it did.
- Keep your explanation to 4 sentences max.
"""
      # Store the image in the first user message
      st.session_state.messages.append({"role": "user", "parts": [prompt, img]})
      
      with st.spinner("Generating explanation..."):
          response = gemini_model.generate_content(st.session_state.messages)
          st.session_state.messages.append({"role": "model", "parts": [response.text]})

  # Display chat messages (skip the initial user prompt since it contains the hidden system instructions)
  for msg in st.session_state.messages:
      if msg["role"] == "user":
          if msg == st.session_state.messages[0]:
              continue
          with st.chat_message("user"):
              st.markdown(msg["parts"][0])
      else:
          with st.chat_message("assistant"):
              st.markdown(msg["parts"][0])

  # Accept user input
  if prompt := st.chat_input("Ask a follow-up question about the MRI scan..."):
      st.session_state.messages.append({"role": "user", "parts": [prompt]})
      with st.chat_message("user"):
          st.markdown(prompt)

      with st.chat_message("assistant"):
          message_placeholder = st.empty()
          with st.spinner("Thinking..."):
              # The Gemini API doesn't support inline images in the chat history.
              # We compile the history into a single text prompt and send the image alongside it.
              img = PIL.Image.open(saliency_map_path)
              full_context = ""
              for msg in st.session_state.messages:
                  if msg["role"] == "user":
                      full_context += f"User: {msg['parts'][0]}\n\n"
                  else:
                      full_context += f"Model: {msg['parts'][0]}\n\n"
              
              response = gemini_model.generate_content([full_context, img])
              message_placeholder.markdown(response.text)
          st.session_state.messages.append({"role": "model", "parts": [response.text]})
