<div align="center">

# 🏥 WellScan Healthcare

### AI-Powered Brain Tumor Detection & Segmentation with Blockchain Medical Records

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Next.js](https://img.shields.io/badge/Next.js-Frontend-000000?logo=nextdotjs&logoColor=white)](https://nextjs.org)
[![Ethereum](https://img.shields.io/badge/Ethereum-Blockchain-3C3C3D?logo=ethereum&logoColor=white)](https://ethereum.org)

</div>

---

## 📌 Problem Statement

Brain tumors require fast, accurate diagnosis — yet manual MRI analysis is time-consuming and specialist-dependent. Patient medical records are siloed across hospitals, leading to data loss and lack of privacy. WellScan addresses both problems by combining **deep learning-based tumor detection** with a **blockchain-secured medical records system**.

---

## 🧠 What It Does

| Feature | Description |
|---|---|
| **Tumor Classification** | Classifies MRI scans into 4 classes: Meningioma, Glioma, Pituitary Tumor, No Tumor |
| **Tumor Segmentation** | Highlights the exact tumor region using pixel-level segmentation |
| **Bounding Box Detection** | Draws bounding box around detected tumor location |
| **Blockchain Records** | Stores patient & doctor records on Ethereum — tamper-proof & decentralized |
| **Doctor/Patient Portal** | Registration, appointments, and diagnosis history via a Next.js frontend |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       WellScan System                        │
├─────────────────┬──────────────────┬────────────────────────┤
│   AI Backend    │   Node Server    │   Blockchain Frontend  │
│   (Flask/Python)│   (Express/Mongo)│   (Next.js/Ethereum)   │
│                 │                  │                        │
│  AT-RES-U-Net   │  Doctor/Patient  │  Smart Contract        │
│  (Custom CNN)   │  Registration    │  (Solidity/Record.sol) │
│  Classification │  Auth & Storage  │  MetaMask Integration  │
│  + Segmentation │  (MongoDB)       │  Rinkeby Testnet       │
└─────────────────┴──────────────────┴────────────────────────┘
```

### Model: AT-RES-U-Net
- **Dual-head output**: Classification head (4 classes) + Segmentation head (pixel mask)
- **Custom loss metrics**: Mean IoU and Dice Coefficient
- **Preprocessing**: Otsu thresholding → contour cropping → z-score normalization
- **Image size**: 224×224 grayscale

---

## 🖼️ Sample Results

| Input MRI | Tumor Detection | Segmentation Mask |
|:---------:|:---------------:|:-----------------:|
| ![Input](docs/screenshots/sample_mri_input.jpg) | ![Detected](docs/screenshots/sample_output_detection.jpg) | ![Segmented](docs/screenshots/sample_output_segmentation.jpg) |

---

## 🛠️ Tech Stack

**AI/ML:** Python · TensorFlow 2.15 · Keras · OpenCV · NumPy · SciPy · Flask  
**Backend:** Node.js · Express.js · MongoDB · Mongoose  
**Frontend/Blockchain:** Next.js · Ethereum · Solidity · Web3.js · MetaMask  

---

## 🚀 Running Locally (3 Services)

### Prerequisites
- Python 3.10, Node.js 18+, MongoDB, MetaMask browser extension
- Download model weights: `AT_RESu_net_all_STD_0.0927_th=0.38.hdf5`  
  *(Too large for GitHub — [Download from Google Drive](YOUR_DRIVE_LINK_HERE))*

---

### 1️⃣ AI Backend (Flask — Port 5000)

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Place the .hdf5 model file in the backend/ folder, then:
MODEL_PATH=AT_RESu_net_all_STD_0.0927_th=0.38.hdf5 python app.py
```

**Test the API:**
```bash
curl -X POST http://localhost:5000/predict \
  -F "file=@docs/screenshots/sample_mri_input.jpg"
```

Expected response:
```json
{
  "prediction": "glioma 97.43%",
  "original_img": "/static/uploads/bbox_sample_mri_input.jpg",
  "segmented_img": "/static/uploads/segmented_sample_mri_input.jpg"
}
```

---

### 2️⃣ Node Auth Server (Express — Port 8080)

```bash
cd server
npm install
# Make sure MongoDB is running locally (mongod)
node index.js
```

---

### 3️⃣ Blockchain Frontend (Next.js — Port 3000)

```bash
cd blockchain
npm install

# Deploy smart contract (only needed once):
cd ethereum
node compile.js
node deploy.js   # Copy the deployed contract address

cd ..
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.  
Make sure MetaMask is connected to **Rinkeby Test Network**.

---

## 📁 Project Structure

```
wellscanhealthcare/
├── backend/                    # Flask AI inference server
│   ├── app.py                  # Main Flask app (fixed, no hardcoded paths)
│   ├── requirements.txt
│   └── static/uploads/         # Runtime: stores predictions
│
├── server/                     # Express.js auth & registration server
│   ├── index.js                # REST API (doctors/patients CRUD)
│   └── package.json
│
├── blockchain/                 # Next.js frontend + Ethereum smart contracts
│   ├── pages/                  # UI pages (dashboard, registration, diagnosis)
│   ├── components/             # Shared layout & nav
│   ├── ethereum/
│   │   ├── contracts/Record.sol  # Solidity smart contract
│   │   ├── compile.js
│   │   └── deploy.js
│   └── package.json
│
├── docs/screenshots/           # Sample MRI images & outputs
└── README.md
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Segmentation STD | 0.0927 |
| Optimal Threshold | 0.38 |
| Classes | Meningioma, Glioma, Pituitary, No Tumor |

---

## 📚 Dataset

MRI Brain Tumor dataset from [Kaggle — Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

---

## 👨‍💻 Authors

BTech Final Year Project — *[Your College Name]*, 2024  
Developed by [Your Name] & Team

---

## 📄 License

MIT License — free to use for academic and personal projects.
