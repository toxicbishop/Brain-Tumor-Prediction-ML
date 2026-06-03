# 🧠 Brain Tumor Classification Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle&logoColor=white)

## 📖 Overview

This project classifies brain tumors from MRI scans using advanced deep learning techniques. The goal is to assist healthcare professionals in the early detection and diagnosis of brain tumors. The project uses transfer learning with the pre-trained **Xception model** and a custom-built **Convolutional Neural Network (CNN)** to classify MRI images into four categories:

| Class         | Description                                         |
| ------------- | --------------------------------------------------- |
| 🔴 Glioma     | A tumor that starts in the glial cells of the brain |
| 🟠 Meningioma | A tumor arising from meninges surrounding the brain |
| 🟡 Pituitary  | A tumor on the pituitary gland                      |
| 🟢 No Tumor   | No tumor detected                                   |

An interactive **Streamlit** web app allows users to upload MRI scans and receive real-time predictions with visual saliency map explanations powered by **Gemini 1.5 Flash**.

---

## 📸 Screenshots

<table>
  <tr>
    <td><img src="images/1.png" alt="Streamlit App - Upload" width="100%"/></td>
    <td><img src="images/2.png" alt="Streamlit App - Results" width="100%"/></td>
  </tr>
  <tr>
    <td align="center">Upload MRI Scan</td>
    <td align="center">Prediction Results + Saliency Map</td>
  </tr>
</table>

![Streamlit Web App - Explanation](images/3.png)

---

## 🌟 Features

### 🧠 Deep Learning Models

- **Xception (Transfer Learning)**
  - Pre-trained on ImageNet, fine-tuned for brain tumor classification
  - Uses depthwise separable convolutions for efficient feature extraction
  - Achieves **99% accuracy** on test data

- **Custom CNN**
  - Built from scratch with Keras — 4 conv blocks + dense layers
  - L2 regularization and dropout for generalization
  - Achieves **98% accuracy** on test data

### 🌐 Interactive Streamlit Web App

- Upload any brain MRI image (JPG/PNG)
- Choose between Xception or Custom CNN model
- See prediction confidence for all 4 classes via interactive Plotly charts

### 🔍 Saliency Map Visualizations

- Gradient-based saliency maps highlight the brain regions the model focuses on
- **Gemini 1.5 Flash** generates a natural-language explanation of each prediction

### 📊 Training & Evaluation

- Training/validation curves for accuracy, loss, precision, and recall
- Confusion matrix with class-name labels
- Full classification report (precision, recall, F1-score per class)
- ROC/AUC curves for multi-class evaluation

---

## 📂 Directory Structure

```plaintext
Brain-Tumor-Prediction-ML/
├── Training/                        # Training MRI dataset (Git LFS)
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
├── Testing/                         # Testing MRI dataset (Git LFS)
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
├── images/                          # README screenshots
├── braintumorclassification.py      # Main training & evaluation script
├── app.py                           # Streamlit web application
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variable template
├── .gitattributes                   # Git LFS configuration
├── .gitignore                       # Files excluded from version control
├── LICENSE                          # MIT License
└── README.md                        # Project documentation
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Brain-Tumor-Prediction-ML.git
cd Brain-Tumor-Prediction-ML
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

```bash
# Copy the example and fill in your keys
copy .env.example .env     # Windows
cp .env.example .env       # macOS/Linux
```

Edit `.env` and add your `GOOGLE_API_KEY` (for Gemini explanations) and optionally `NGROK_AUTH_TOKEN`.

### 5. Download the Dataset

The dataset is auto-downloaded from Kaggle when you run the script.  
Make sure you have a Kaggle API token at `~/.kaggle/kaggle.json`.  
Or manually download from: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

### 6. Train the Models (Optional)

```bash
python braintumorclassification.py
```

### 7. Run the Streamlit Web App

```bash
streamlit run app.py
```

---

## 🧠 Model Architecture

### Xception (Transfer Learning)

```
Xception (ImageNet weights, pooling=max)
    └── Flatten
    └── Dropout(0.3)
    └── Dense(256, relu)
    └── Dropout(0.25)
    └── Dense(4, softmax)
```

### Custom CNN

```
Conv2D(512) → MaxPool →
Conv2D(256) → MaxPool → Dropout(0.25) →
Conv2D(128) → MaxPool → Dropout(0.25) →
Conv2D(64)  → MaxPool →
Flatten → Dense(256, relu, L2) → Dropout(0.35) →
Dense(4, softmax)
```

**Training Config:**

- Optimizer: `Adamax(lr=0.001)`
- Loss: `categorical_crossentropy`
- Callbacks: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`
- Data Augmentation: rotation, shift, flip, zoom, brightness

---

## 📊 Results

| Model                        | Test Accuracy | Test Loss |
| ---------------------------- | ------------- | --------- |
| Xception (Transfer Learning) | **99%**       | ~0.03     |
| Custom CNN                   | **98%**       | ~0.07     |

---

## 🛠️ Tools & Technologies

| Category        | Tools                            |
| --------------- | -------------------------------- |
| Language        | Python 3.12.7+                   |
| Deep Learning   | TensorFlow 2.x, Keras            |
| Data            | NumPy, Pandas, Scikit-learn      |
| Visualization   | Matplotlib, Seaborn, Plotly      |
| Web App         | Streamlit                        |
| AI Explanations | Gemini 1.5 Flash                 |
| Dataset         | Kaggle — Brain Tumor MRI Dataset |
| Deployment      | ngrok (optional)                 |

---

## 🔮 Future Enhancements

- [ ] Grad-CAM visualizations (more interpretable than gradient saliency)
- [ ] Multi-class ROC/AUC curve plots
- [ ] Interactive chat with Gemini about the MRI scan
- [ ] Historical analysis — compare scans over time
- [ ] Mobile-friendly app

---

## ⚠️ Disclaimer

This project is intended for **academic and research purposes only**. It is not a medical device and should not be used as a substitute for professional medical diagnosis.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
