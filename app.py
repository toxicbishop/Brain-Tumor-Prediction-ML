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

def generate_grad_cam(model, img_array, class_index, img_size, img, uploaded_file):
    is_nested = isinstance(model.layers[0], tf.keras.Model)
    target_layer = get_target_layer(model)
    
    with tf.GradientTape() as tape:
        if is_nested:
            base_model = model.layers[0]
            base_output_tensor = base_model.outputs[0] if hasattr(base_model, 'outputs') and base_model.outputs else base_model.layers[-1].output
            base_grad_model = tf.keras.Model(base_model.inputs, [target_layer.output, base_output_tensor])
            conv_outputs, base_outputs = base_grad_model(img_array)
            
            tape.watch(conv_outputs)
            
            x = base_outputs
            for layer in model.layers[1:]:
                x = layer(x)
            predictions = x
        else:
            x = img_array
            conv_outputs = None
            for layer in model.layers:
                x = layer(x)
                if layer.name == target_layer.name:
                    conv_outputs = x
                    tape.watch(conv_outputs)
            predictions = x
            
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
    
    original_img = image.img_to_array(img)
    superimposed_img = heatmap * 0.4 + original_img * 0.6
    superimposed_img = superimposed_img.astype(np.uint8)
    
    saliency_map_path = os.path.join(output_dir, uploaded_file.name)
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

def main():
    st.title("Brain Tumor Classification")
    st.write("Upload an image of a brain MRI scan to classify")

    analysis_mode = st.radio("Analysis Mode", ["Single Scan Analysis", "Historical Comparison"])

    selected_model = st.radio(
        "Selected Model",
        ("Transfer Learning - Xception", "Custom CNN", "Ensemble (Xception + CNN)")
    )

    if selected_model == "Transfer Learning - Xception":
        model = load_xception_model('xception_model.weights.h5')
        img_size_default = (299, 299)
    elif selected_model == "Custom CNN":
        model = load_model('cnn_model.h5')
        img_size_default = (224, 224)
    else:
        model_xc = load_xception_model('xception_model.weights.h5')
        model_cnn = load_model('cnn_model.h5')
        img_size_default = (299, 299)
        model = None # Used for saliency generation later

    labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

    def process_scan(uploaded_file):
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
        
            target_model = model_xc
            img_size = (299, 299)
            img_array = img_array_xc
            img = img_xc
        else:
            img = image.load_img(uploaded_file, target_size=img_size_default)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            prediction = model.predict(img_array)
            target_model = model
            img_size = img_size_default

        class_index = np.argmax(prediction[0])
        result = labels[class_index]
    
        saliency_map = generate_grad_cam(target_model, img_array, class_index, img_size, img, uploaded_file)
        saliency_map_path = f'saliency_maps/{uploaded_file.name}'
    
        return {
            'prediction': prediction,
            'class_index': class_index,
            'result': result,
            'saliency_map': saliency_map,
            'saliency_map_path': saliency_map_path
        }

    def display_results(results, uploaded_file, title="Classification Results"):
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
            st.image(uploaded_file, caption=f"Uploaded Image: {title}", use_container_width=True)

        with col2:
            st.image(results['saliency_map'], caption=f"Grad-CAM Heatmap: {title}", use_container_width=True)

        st.write(f"## {title}")

        result_container = st.container()
        result_container.markdown(
            f"""
            <div style="background-color: #000000; color: #ffffff; padding: 30px; border-radius: 15px;">
              <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="flex: 1; text-align: center;">
                  <h3 style="color: #ffffff; margin-bottom: 10px; font-size: 20px;">Prediction</h3>
                  <p style="font-size: 36px; font-weight: 800; color: #FF0000; margin: 0;">
                    {results['result']}
                  </p>
                </div>
                <div style="width: 2px; height: 80px; background-color: #ffffff; margin: 0 20px;"></div>
                <div style="flex: 1; text-align: center;">
                  <h3 style="color: #ffffff; margin-bottom: 10px; font-size: 20px;">Confidence</h3>
                  <p style="font-size: 36px; font-weight: 800; color: #2196F3; margin: 0;">
                    {results['prediction'][0][results['class_index']]:.4%}
                  </p>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True)

        probabilities = results['prediction'][0]
        sorted_indices = np.argsort(probabilities)[::-1]
        sorted_labels = [labels[i] for i in sorted_indices]
        sorted_probabilities = probabilities[sorted_indices]

        fig = go.Figure(go.Bar(
            x=sorted_probabilities,
            y=sorted_labels,
            orientation='h',
            marker_color=['red' if label == results['result'] else 'blue' for label in sorted_labels])
        )

        fig.update_layout(
            title=f'Probabilities for {title}',
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


    gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash") # type: ignore

    if analysis_mode == "Single Scan Analysis":
        uploaded_file = st.file_uploader("Choose an Image...", type=["jpg", "jpeg", "png"])
    
        if uploaded_file is not None:
            results = process_scan(uploaded_file)
            display_results(results, uploaded_file)
        
            st.write("## Interactive Explanation")
        
            if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
                st.session_state.current_file = uploaded_file.name
                st.session_state.messages = []
            
            if len(st.session_state.messages) == 0:
                img = PIL.Image.open(results['saliency_map_path'])
                prompt = f"""You are an expert neurologist. You are tasked with explaining a Grad-CAM heatmap of a brain tumor MRI scan. The heatmap was generated by a deep learning model that was trained to classify brain tumors as either glioma, meningioma, pituitary, or no tumor.

    The heatmap highlights the regions of the image that the machine learning model is focusing on to make the prediction.

    The deep learning model predicted the image to be of class '{results['result']}' with a confidence of {results['prediction'][0][results['class_index']] * 100}%.

    In your response:
    - Explain what regions of the brain the model is focusing on, based on the heatmap (red/yellow areas represent high focus).
    - Explain possible reasons why the model made the prediction it did.
    - Keep your explanation to 4 sentences max.
    """
                st.session_state.messages.append({"role": "user", "parts": [prompt, img]})
            
                with st.spinner("Generating explanation..."):
                    response = gemini_model.generate_content(st.session_state.messages)
                    st.session_state.messages.append({"role": "model", "parts": [response.text]})

            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    if msg == st.session_state.messages[0]:
                        continue
                    with st.chat_message("user"):
                        st.markdown(msg["parts"][0])
                else:
                    with st.chat_message("assistant"):
                        st.markdown(msg["parts"][0])

            if prompt := st.chat_input("Ask a follow-up question about the MRI scan..."):
                st.session_state.messages.append({"role": "user", "parts": [prompt]})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    with st.spinner("Thinking..."):
                        img = PIL.Image.open(results['saliency_map_path'])
                        full_context = ""
                        for msg in st.session_state.messages:
                            if msg["role"] == "user":
                                full_context += f"User: {msg['parts'][0]}\\n\\n"
                            else:
                                full_context += f"Model: {msg['parts'][0]}\\n\\n"
                    
                        response = gemini_model.generate_content([full_context, img])
                        message_placeholder.markdown(response.text)
                    st.session_state.messages.append({"role": "model", "parts": [response.text]})

    else:
        # Historical Comparison
        col_prev, col_curr = st.columns(2)
        with col_prev:
            prev_file = st.file_uploader("Upload Previous Scan (Older)", type=["jpg", "jpeg", "png"], key='prev')
        with col_curr:
            curr_file = st.file_uploader("Upload Recent Scan (Newer)", type=["jpg", "jpeg", "png"], key='curr')
        
        if prev_file is not None and curr_file is not None:
            st.write("---")
            results_prev = process_scan(prev_file)
            results_curr = process_scan(curr_file)
        
            display_results(results_prev, prev_file, title="Previous Scan")
            display_results(results_curr, curr_file, title="Recent Scan")
        
            st.write("## Historical Comparative Analysis")
            session_key = f"{prev_file.name}_{curr_file.name}"
        
            if "current_file" not in st.session_state or st.session_state.current_file != session_key:
                st.session_state.current_file = session_key
                st.session_state.messages = []
            
            if len(st.session_state.messages) == 0:
                img_prev = PIL.Image.open(results_prev['saliency_map_path'])
                img_curr = PIL.Image.open(results_curr['saliency_map_path'])
            
                prompt = f"""You are an expert oncologist. You are tasked with comparing two brain MRI scans from a patient over time.
    The first image is the PREVIOUS scan. The second image is the RECENT scan.
    Both images have Grad-CAM heatmaps overlaid (red/yellow areas indicate model focus).

    Previous Scan Prediction: {results_prev['result']} ({results_prev['prediction'][0][results_prev['class_index']] * 100:.2f}% confidence)
    Recent Scan Prediction: {results_curr['result']} ({results_curr['prediction'][0][results_curr['class_index']] * 100:.2f}% confidence)

    In your response:
    - Compare the two scans and their heatmaps. 
    - Is the tumor growing, shrinking, or stable? Did the classification change?
    - Explain the clinical significance of these changes.
    - Keep your explanation concise and professional (4-5 sentences).
    """
                st.session_state.messages.append({"role": "user", "parts": [prompt, img_prev, img_curr]})
            
                with st.spinner("Generating comparative analysis..."):
                    response = gemini_model.generate_content(st.session_state.messages)
                    st.session_state.messages.append({"role": "model", "parts": [response.text]})

            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    if msg == st.session_state.messages[0]:
                        continue
                    with st.chat_message("user"):
                        st.markdown(msg["parts"][0])
                else:
                    with st.chat_message("assistant"):
                        st.markdown(msg["parts"][0])

            if prompt := st.chat_input("Ask a follow-up question about the scan comparison..."):
                st.session_state.messages.append({"role": "user", "parts": [prompt]})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    with st.spinner("Thinking..."):
                        img_prev = PIL.Image.open(results_prev['saliency_map_path'])
                        img_curr = PIL.Image.open(results_curr['saliency_map_path'])
                        full_context = ""
                        for msg in st.session_state.messages:
                            if msg["role"] == "user":
                                full_context += f"User: {msg['parts'][0]}\\n\\n"
                            else:
                                full_context += f"Model: {msg['parts'][0]}\\n\\n"
                    
                        response = gemini_model.generate_content([full_context, img_prev, img_curr])
                        message_placeholder.markdown(response.text)
                    st.session_state.messages.append({"role": "model", "parts": [response.text]})

if __name__ == "__main__":
    main()
