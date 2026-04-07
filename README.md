# Satellite LULC Classification & Change Detection

A high-performance pipeline for Land Use / Land Cover (LULC) classification using Sentinel-2 multispectral imagery and ESA WorldCover labels. This project is specifically optimized for localized high-accuracy urban mapping, featuring advanced architectural upgrades like ResUNet and Test-Time Augmentation (TTA).

---

## 🚀 Key Features

### 1. High-Accuracy ResUNet Architecture
The core model is a **Residual U-Net (ResUNet)**. By incorporating residual skip connections, the model preserves fine-scale geometric details (like bridges and narrow streets) that are often lost in standard U-Net architectures.

### 2. Multi-Spectral Deep Stack (6-Channel)
Unlike standard RGB models, this pipeline utilizes a 6-band input stack:
- **BANDS:** [Blue, Green, Red, Near-Infrared (NIR), Short-Wave Infrared (SWIR), NDVI]
- **Advantage:** The inclusion of SWIR light provides "spectral X-ray" capabilities, allowing the model to unequivocally distinguish between urban concrete (Built-up) and natural barren soil.

### 3. Advanced Inference Strategies
- **Overlapping Sliding Window:** Eliminates "grid-line" artifacts in large GeoTIFF mosaics.
- **Test-Time Augmentation (TTA):** Predicts on 3 views (Original, Horizontal Flip, Vertical Flip) and averages the probabilities for maximum boundary stability.
- **Bridge Rescue Logic:** A custom heuristic that identifies pixels with ≥30% Built-up probability over water, specifically designed to preserve thin linear infrastructure like bridges.

### 4. Direct ESA S3 Integration
Automated download of ground-truth labels directly from the ESA WorldCover S3 buckets, ensuring perfect alignment with Sentinel-2 tiles without requiring manual GIS work.

---

## 🛠 Technology Stack
- **Framework:** TensorFlow / Keras (Optimized with `tensorflow-metal` for Apple Silicon)
- **Geospatial:** Rasterio, GeoPandas, Shapely
- **Analysis:** NumPy, SciPy (Median filtering & Morphological smoothing)
- **Visualization:** Matplotlib with custom embedded GeoTIFF colormaps

---

## 📥 Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd changedetection2
   ```

2. **Setup Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

---

## 🏗 Usage Pipeline

The project is orchestrated via `main.py`. You can run the entire pipeline or specific stages.

### Stage 1: Data Download
Downloads Sentinel-2 imagery and aligned ESA WorldCover labels for your configured Area of Interest (AOI).
```bash
python main.py --download
```

### Stage 2: Preprocessing
Extracts multispectral patches (256x256) and prepares the 6-channel tensors for training.
```bash
python main.py --preprocess
```

### Stage 3: Training
Trains the ResUNet model using a combination of **Categorical Focal Loss** and **Dice Loss** to handle class imbalance.
```bash
python main.py --train
```

### Stage 4: Prediction (The "TTA" Mode)
Generates high-resolution colored GeoTIFFs of your AOI.
```bash
python main.py --predict
```

---

## 🗺 Land Cover Classes
The output is color-coded according to the following scheme:
- 🟢 **Vegetation:** Green
- 🌲 **Dense Canopy:** Dark Green
- 🔴 **Built-up:** Red
- 🟤 **Barren Land:** Brown
- 🔵 **Water:** Blue

---

## 📂 Project Structure
- `config.py`: Central configuration for AOIs, paths, and hyperparameters.
- `model.py`: ResUNet architecture definition.
- `train.py`: Training logic and custom loss functions.
- `predict.py`: Sliding window, TTA, and Bridge Rescue logic.
- `data_download.py`: S3 Sentinel/WorldCover fetcher.
- `preprocessing.py`: Feature engineering and patch extraction.

---

## 📈 Performance Goals
The pipeline is currently tuned for a **Localized Pixel Accuracy of >80%** and **Mean IoU of >60%**, which represents the technical ceiling for 10m/pixel automated LULC classification using Sentinel-2.
