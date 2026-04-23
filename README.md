# Satellite LULC Classification & Change Detection

A high-performance pipeline for Land Use / Land Cover (LULC) classification and temporal Change Detection using Sentinel-2 multispectral imagery and ESA WorldCover labels. This project is optimized for localized high-accuracy urban and ecological mapping, featuring advanced neural networks, morphological overrides, and automated multi-file processing.

---

## 🚀 Key Features

### 1. High-Accuracy ResUNet Architecture
The core model is a **Residual U-Net (ResUNet)**. By incorporating residual skip connections, the model preserves fine-scale geometric details (like bridges and narrow streets) that are often lost in standard U-Net architectures.

### 2. Multi-Spectral Deep Stack (6-Channel)
Unlike standard RGB models, this pipeline utilizes a 6-band input stack:
- **BANDS:** [Blue, Green, Red, Near-Infrared (NIR), Short-Wave Infrared (SWIR), NDVI]
- **Advantage:** The inclusion of SWIR light provides "spectral X-ray" capabilities, allowing the model to unequivocally distinguish between complex urban and natural targets.

### 3. Automated Sub-Agent Operations
- **Metadata AOI Extraction:** Eliminates hardcoded coordinates by dynamically reading geographic bounding boxes directly from the provided Sentinel-2 GeoTIFF metadata.
- **Overlapping Sliding Window:** Eliminates "grid-line" artifacts in large geographically boundless GeoTIFF mosaics.
- **Bridge Rescue Logic:** A custom spectral heuristic relying on SWIR/NDBI combinations to actively rescue thin linear infrastructure (bridges) misclassified functionally as water.
- **Standalone Barren Land Override:** A strict morphological sequence applying dynamic pixel equations (BSI, NDVI, NDBI) to cleanly bypass neural network logic and overlay absolute Barren Land classifications securely onto map predictions.

### 4. Advanced Change Detection Pipeline
Dedicated routines dynamically compare baseline classifications (e.g., 2020) against updated sets (e.g., 2025) to map out transition matrices like Urban Expansion, Canopy Deforestation, and Ecological Recovery sequences.

---

## 🛠 Technology Stack
- **Framework:** TensorFlow / Keras (Optimized for deep learning model integration)
- **Geospatial:** Rasterio, GeoPandas, Shapely
- **Analysis:** NumPy, SciPy (Median filtering, mathematical thresholding & Morphological operations)
- **Visualization:** Matplotlib with custom embedded GeoTIFF colormaps and transition graphics

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
Downloads Sentinel-2 imagery and aligned ESA WorldCover labels covering your `.tif` metadata limits.
```bash
python main.py --download
```

### Stage 2: Preprocessing
Extracts multispectral patches (256x256), applies categorical index normalizations, and prepares the 6-channel tensors.
```bash
python main.py --preprocess
```

### Stage 3: Training
Trains the ResUNet model using a combination of **Categorical Focal Loss** and **Dice Loss** to handle dataset class imbalances naturally.
```bash
python main.py --train
```

### Stage 4: Prediction
Generates high-resolution colored GeoTIFF predictions of your selected region, including ML and algorithmic Barren/Bridge overrides.
```bash
python main.py --predict
```

### Stage 5: Change Detection
Computes the pixel-level transition matrix between multiple predictive years to visually chart geographic progression elements.
```bash
python change_detection.py
```

### Stage 6: Batch Automation
Automatically runs prediction on a directory full of `.tif` files, preserving the folder structure (ideal for multiple regions like Pune, Palghar, etc.).

**1. Predict on all .tif files in a folder (and its subfolders):**
```bash
python batch_predict.py --input data --output outputs/predictions
```

**2. Run Change Detection on the predicted regions:**
Once the predictions are generated in `outputs/predictions`, you can run change detection to analyze the built-up percent change and area increase.
For example, if you have predictions for 2020 and 2025 images:
```bash
python change_detection.py
```
*(If you need to batch process change detection for multiple region folders like Pune and Palghar, you can use the batch change detection script if available).*

---

## 🗺 Land Cover Classes
The model and scripts operate using a hyper-resolved 5-Class system:
- 🟢 **Vegetation:** Green (Grasses, Shrubs)
- 🌲 **Dense Canopy:** Dark Green (Trees, Forests, Mangroves)
- 🔴 **Built-up:** Red (Urban grids, Roads)
- 🔵 **Water:** Blue (Deep pools, Lakes)
- 🟤 **Barren Land:** Brown (Bare earth, Sparse ground, Sand)

---

## 📂 Project Structure
- `config.py`: Central configuration for input paths, color mappings, and hyperparameters.
- `model.py`: ResUNet architecture definition.
- `train.py`: Training logic and custom class-weighted loss schemas.
- `predict.py`: Sliding window inference alongside spectral heuristics algorithms (Bridges & Barren intercepts).
- `preprocessing.py`: Feature engineering bounding boxes and multi-spectral feature stacking schemas.
- `change_detection.py`: Mathematical sequence maps calculating chronological ecosystem transitions.
- `batch_predict.py`: Iterator orchestrating high-volume regional dataset runs.