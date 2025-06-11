# Fruit Ripeness Detection

A machine learning project that detects fruit ripeness using image processing and pattern recognition techniques.

## Features

- **5 ML Algorithms:** SVM, Random Forest, Gradient Boosting, Logistic Regression, Ensemble
- **Image Processing:** Color, texture, and shape feature extraction
- **GUI Interface:** User-friendly Tkinter application
- **Model Comparison:** Automatic best model selection

## Quick Start

### 1. Dataset Setup
Download dataset: [@Kaggle - Fruit Image Dataset](https://www.kaggle.com/datasets/mdsagorahmed/fruit-image-dataset-22-classes)

Extract to: `dataset/Ripe & Unripe Fruits/`

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train Model
```bash
python app.py
```

### 4. Test Model
```bash
python test.py
```

### 5. Use GUI
```bash
python UI.py
```

## Project Structure
```
├── app.py           # Model training
├── test.py          # Model testing  
├── UI.py            # GUI interface
├── requirements.txt # Dependencies
└── dataset/         # Image data (download separately)
```

## Requirements
- Python 3.7+
- Libraries: numpy, pandas, scikit-learn, opencv-python, matplotlib, seaborn, scikit-image, Pillow, joblib, tqdm

## Supported Fruits
Apple, Banana, Dragon Fruit, Grapes, Lemon, Mango, Orange, Papaya, Pineapple, Pomegranate, Strawberry

## Results
- Model comparison graphs (accuracy, F1-score, AUC)
- Confusion matrices for each algorithm
- Best model automatically saved as `.pkl` file

## Usage Examples

**Train specific fruit:**
```bash
python app.py apple
```

**Test specific model:**
```bash
python test.py your_model.pkl
```

## Notes
- Training takes 5-15 minutes depending on system
- Dataset not included in repository (large size)
- Works on Windows, macOS, Linux 