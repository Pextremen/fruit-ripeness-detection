"""
Fruit Ripeness Detection - Test Script
This script is used to test trained models.
"""

import os
import sys
import glob
import joblib
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import random
from app import FruitRipenessDetection

def main():
    # Check command line arguments
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Automatically find model files
        model_files = [f for f in os.listdir('.') if f.endswith('_model.pkl') and not f.endswith('_info.pkl')]
        
        if not model_files:
            print("Error: No trained model found.")
            print("You need to train a model first. Run 'python app.py' command.")
            return
        
        model_path = model_files[0]
        print(f"Automatically selected model: {model_path}")
    
    # Check model info file
    info_path = f"{model_path}_info.pkl"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    
    try:
        # Load model
        model = joblib.load(model_path)
        
        # Load model information
        model_info = None
        if os.path.exists(info_path):
            model_info = joblib.load(info_path)
            print("Model information loaded.")
        
        # Create fruit detector
        data_folder = "dataset/Ripe & Unripe Fruits"
        detector = FruitRipenessDetection(data_folder)
        
        # Transfer model information
        if model_info:
            detector.label_names = model_info.get('label_names', np.array(['unripe', 'ripe']))
            detector.feature_names = model_info.get('feature_names', [])
            detector.scaler = model_info.get('scaler', None)
            detector.selector = model_info.get('selector', None)
        else:
            # Default values
            detector.label_names = np.array(['unripe', 'ripe'])
        
        # Set model name (from filename)
        model_name = os.path.basename(model_path).replace('_model.pkl', '').replace('_', ' ').title()
        detector.models[model_name] = model
        
        # Find test images
        test_images = []
        for ripeness in ["ripe", "unripe"]:
            fruit_folders = glob.glob(os.path.join(data_folder, f"{ripeness}*"))
            for folder in fruit_folders:
                images = glob.glob(os.path.join(folder, "*.jpg"))
                # Select maximum 2 random images from each folder
                if images:
                    samples = random.sample(images, min(2, len(images)))
                    test_images.extend(samples)
        
        if not test_images:
            print("Error: No test images found.")
            return
        
        # Select 4 random images
        test_images = random.sample(test_images, min(4, len(test_images)))
        
        # Test and visualize
        plt.figure(figsize=(15, 10))
        
        for i, img_path in enumerate(test_images):
            # Read image
            img = io.imread(img_path)
            
            # Make prediction
            prediction = detector.predict(model_name, img_path)
            
            # Get actual label from image name
            folder_name = os.path.basename(os.path.dirname(img_path))
            actual = folder_name.split()[0]  # "ripe" or "unripe"
            
            # Show image
            plt.subplot(2, 2, i+1)
            plt.imshow(img)
            
            # Is prediction correct?
            is_correct = prediction.lower() == actual.lower()
            color = 'green' if is_correct else 'red'
            
            plt.title(f"Prediction: {prediction}\nActual: {actual}", color=color)
            plt.axis('off')
        
        plt.tight_layout()
        plt.suptitle(f"Model Test: {model_name}", fontsize=16)
        plt.subplots_adjust(top=0.9)
        plt.savefig("model_test_results.png")
        plt.show()
        
        print(f"Test completed. Results saved to 'model_test_results.png'.")
        
    except Exception as e:
        print(f"Error: Problem occurred during testing: {str(e)}")

if __name__ == "__main__":
    main()
