# 🤟 ASL Sign Language Classifier
 
A real-time American Sign Language (ASL) recognition system using a CNN model trained on the Sign MNIST dataset, served via a Flask web app with live webcam feed.
 
![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=flat-square&logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.11-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
 
---
 
## 📸 Demo
 
> Point your hand at the webcam inside the blue ROI box — the model predicts the ASL letter in real time.
 
---
 
## 🗂️ Project Structure
 
```
asl-sign-classifier/
│
├── data/                        # Place Sign MNIST CSV files here
│   ├── sign_mnist_train.csv
│   └── sign_mnist_test.csv
│
├── model/                       # Generated model files (gitignored)
│   ├── asl_model.h5             ← created by train.py
│   └── asl_model_fixed.h5      ← created by fix_model.py
│
├── project/                     # Flask web app
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── style.css
│       └── main.js
│
├── train.py                     # Train the CNN model
├── fix_model.py                 # Patch batch_shape bug in .h5
├── predict_image.py             # Test prediction on a static image
├── requirements.txt
└── README.md
```
 
---
 
## ⚙️ Setup
 
### 1. Clone the repo
 
```bash
git clone https://github.com/lucifer230407/asl-sign-classifier.git
cd asl-sign-classifier
```
 
### 2. Create a conda environment
 
```bash
conda create -n asl310 python=3.10 -y
conda activate asl310
pip install -r requirements.txt
```
 
### 3. Get the dataset
 
Download the [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) dataset from Kaggle and place the CSV files in the `data/` folder:
 
```
data/sign_mnist_train.csv
data/sign_mnist_test.csv
```
 
---
 
## 🚀 Usage
 
### Step 1 — Train the model
 
```bash
python train.py
```
 
Trains a CNN on the Sign MNIST dataset and saves `model/asl_model.h5`. Uses early stopping and data augmentation. Expects ~95%+ validation accuracy.
 
### Step 2 — Fix the model
 
```bash
python fix_model.py
```
 
Patches a `batch_shape` compatibility issue in the `.h5` file for `tf_keras`. Creates `model/asl_model_fixed.h5`.
 
### Step 3 — Test on a static image (optional)
 
```bash
python predict_image.py path/to/image.png
```
 
### Step 4 — Run the web app
 
```bash
cd project
python app.py
```
 
Open `http://127.0.0.1:5000` in your browser.
 
---
 
## 🧠 Model Architecture
 
| Layer | Details |
|-------|---------|
| Input | 28×28×1 grayscale |
| Conv2D × 2 | 32 filters, BatchNorm, MaxPool, Dropout |
| Conv2D × 2 | 64 filters, BatchNorm, MaxPool, Dropout |
| Conv2D × 1 | 128 filters, BatchNorm, MaxPool, Dropout |
| Dense | 256 units, BatchNorm, Dropout 0.5 |
| Output | 24 classes (softmax) |
 
- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy
- **Augmentation:** Rotation, zoom, shifts
- **Classes:** A–Y excluding J and Z (motion signs)
 
---
 
## 🌐 Web App Features
 
- 📷 Live webcam stream via MJPEG
- 🔤 Real-time letter prediction with confidence score
- 🏆 Top 3 candidate letters with probability bars
- 🔡 Interactive alphabet grid highlighting the detected letter
- 🧮 Majority-vote smoothing over last 10 frames to reduce flickering
- ⚡ 15 FPS inference cap for performance
 
---
 
## 📦 Tech Stack
 
| Tool | Purpose |
|------|---------|
| TensorFlow 2.13 + tf-keras | Model training & inference |
| OpenCV | Webcam capture & frame processing |
| Flask | Web server & video streaming |
| NumPy | Array operations |
| Pandas | Dataset loading |
| Matplotlib | Training plots |
 
---
 
## 📁 Notes
 
- Model `.h5` files are excluded from git (too large). Train locally using the steps above.
- CSV data files are also excluded. Download from Kaggle.
- Tested on macOS ARM (M1/M2) with Python 3.10 via conda.
 
---
 
## 👤 Author
 
**Himanshu Jangra**
- GitHub: [@lucifer230407](https://github.com/lucifer230407)
- LinkedIn: [himanshu-jangra](https://linkedin.com/in/himanshu-jangra-933385324)
 





