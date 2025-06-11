"""
Fruit Ripeness Detection - Pattern Recognition Project
Preprocessing, Feature Extraction, Feature Selection and Classifier Implementation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, roc_curve, auc, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage import io, color, img_as_ubyte
import warnings
warnings.filterwarnings('ignore')
import glob
from tqdm import tqdm
import sys

class FruitRipenessDetection:
    def __init__(self, data_folder):
        """
        Fruit ripeness detection class
        
        Args:
            data_folder (str): Folder containing the images
        """
        self.data_folder = data_folder
        self.X = None
        self.y = None
        self.fruit_types = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.feature_names = []
        self.scaler = None
        self.selector = None
        
    def load_data(self, selected_fruit=None):
        """
        Load images from data folder
        
        Args:
            selected_fruit (str, optional): To load specific fruit type
            
        Returns:
            tuple: Features and labels
        """
        print("Loading data...")
        
        # List all subfolders in dataset
        fruit_folders = os.listdir(self.data_folder)
        
        features = []
        labels = []
        fruit_types = []
        
        # Find ripe and unripe categories
        for fruit_folder in fruit_folders:
            fruit_path = os.path.join(self.data_folder, fruit_folder)
            
            # Process only if it's a directory
            if os.path.isdir(fruit_path):
                # Parse folder name (ripe apple, unripe banana etc.)
                parts = fruit_folder.split()
                if len(parts) < 2:
                    continue
                    
                ripeness = parts[0]  # ripe or unripe
                fruit_type = " ".join(parts[1:])  # apple, banana, etc.
                
                # Skip if specific fruit selected and this is not it
                if selected_fruit and fruit_type != selected_fruit:
                    continue
                
                print(f"Processing {ripeness} {fruit_type}...")
                
                # Find images
                image_paths = glob.glob(os.path.join(fruit_path, "*.jpg")) + \
                             glob.glob(os.path.join(fruit_path, "*.png"))
                
                print(f"Found {len(image_paths)} images in {ripeness} {fruit_type} category.")
                
                for image_path in tqdm(image_paths):
                    try:
                        # Load image
                        img = io.imread(image_path)
                        
                        # Feature extraction
                        features_vector = self.extract_features(img)
                        
                        if features_vector is not None:
                            features.append(features_vector)
                            labels.append(ripeness)  # ripe or unripe
                            fruit_types.append(fruit_type)
                    except Exception as e:
                        print(f"Error: Problem occurred while processing {image_path}: {e}")
        
        self.X = np.array(features)
        self.y = np.array(labels)
        self.fruit_types = np.array(fruit_types)
        
        # Encode labels
        le = LabelEncoder()
        self.y = le.fit_transform(self.y)
        self.label_names = le.classes_
        
        # Encode fruit types as well
        le_fruit = LabelEncoder()
        self.fruit_types_encoded = le_fruit.fit_transform(self.fruit_types)
        self.fruit_type_names = le_fruit.classes_
        
        print(f"Total {len(self.X)} samples loaded, {len(np.unique(self.y))} different ripeness classes found.")
        print(f"Label names: {self.label_names}")
        print(f"Fruit types: {self.fruit_type_names}")
        
        return self.X, self.y
        
    def extract_features(self, img):
        """
        Extract features from image
        
        Args:
            img: Image
            
        Returns:
            array: Extracted features
        """
        # Initial check: image dimensions
        if img is None or img.size == 0:
            return None
        
        # Convert image to RGB (if not already)
        if len(img.shape) == 2:  # If grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # If RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        # Resize
        img = cv2.resize(img, (100, 100))
        
        # 1. Color features (RGB mean and standard deviation)
        r_mean, g_mean, b_mean = np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2])
        r_std, g_std, b_std = np.std(img[:,:,0]), np.std(img[:,:,1]), np.std(img[:,:,2])
        
        # 2. HSV color space features
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h_mean, s_mean, v_mean = np.mean(hsv_img[:,:,0]), np.mean(hsv_img[:,:,1]), np.mean(hsv_img[:,:,2])
        h_std, s_std, v_std = np.std(hsv_img[:,:,0]), np.std(hsv_img[:,:,1]), np.std(hsv_img[:,:,2])
        
        # 3. LAB color space features (important for fruits)
        lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l_mean, a_mean, b_mean_lab = np.mean(lab_img[:,:,0]), np.mean(lab_img[:,:,1]), np.mean(lab_img[:,:,2])
        l_std, a_std, b_std_lab = np.std(lab_img[:,:,0]), np.std(lab_img[:,:,1]), np.std(lab_img[:,:,2])
        
        # 4. Texture features (GLCM)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = img_as_ubyte(gray)
        
        # Create GLCM matrix
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(gray, distances, angles, 256, symmetric=True, normed=True)
        
        # Calculate GLCM properties
        contrast = graycoprops(glcm, 'contrast').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        
        # 5. Shape features
        # Convert image to binary (thresholding)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Get the largest contour (assume it's the fruit)
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            
            # Perimeter/Area ratio (compactness)
            if area > 0:
                compactness = perimeter ** 2 / area
            else:
                compactness = 0
                
            # Aspect ratio
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Extent (object area / bounding rectangle area)
            rect_area = w * h
            extent = float(area) / rect_area if rect_area > 0 else 0
            
            # Solidity (object area / convex hull area)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
        else:
            compactness = 0
            aspect_ratio = 0
            extent = 0
            solidity = 0
            area = 0
            perimeter = 0
        
        # Create feature names if not exists
        if not self.feature_names:
            self.feature_names = [
                'r_mean', 'g_mean', 'b_mean', 'r_std', 'g_std', 'b_std',
                'h_mean', 's_mean', 'v_mean', 'h_std', 's_std', 'v_std',
                'l_mean', 'a_mean', 'b_mean_lab', 'l_std', 'a_std', 'b_std_lab',
                'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation',
                'compactness', 'aspect_ratio', 'extent', 'solidity', 'area', 'perimeter'
            ]
        
        # Combine all features
        features = [
            r_mean, g_mean, b_mean, r_std, g_std, b_std,
            h_mean, s_mean, v_mean, h_std, s_std, v_std,
            l_mean, a_mean, b_mean_lab, l_std, a_std, b_std_lab,
            contrast, dissimilarity, homogeneity, energy, correlation,
            compactness, aspect_ratio, extent, solidity, area, perimeter
        ]
        
        return np.array(features)
        
    def preprocess_data(self):
        """
        Preprocess data: normalization and feature scaling
        """
        print("Preprocessing data...")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        
        # Feature scaling
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print("Feature scaling completed.")
        
    def feature_selection(self, k=15):
        """
        Select best features using SelectKBest
        
        Args:
            k (int): Number of features to select
        """
        print(f"Selecting best {k} features...")
        
        self.selector = SelectKBest(score_func=f_classif, k=k)
        self.X_train = self.selector.fit_transform(self.X_train, self.y_train)
        self.X_test = self.selector.transform(self.X_test)
        
        # Get selected feature names
        selected_indices = self.selector.get_support(indices=True)
        selected_features = [self.feature_names[i] for i in selected_indices]
        
        print(f"Selected features: {selected_features}")
        
    def create_models(self):
        """
        Create classification models
        """
        print("Creating models...")
        
        # SVM with RBF kernel
        self.models['SVM'] = SVC(kernel='rbf', probability=True, random_state=42)
        
        # Random Forest
        self.models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Gradient Boosting
        self.models['Gradient Boosting'] = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        # Logistic Regression
        self.models['Logistic Regression'] = LogisticRegression(random_state=42, max_iter=1000)
        
        # Ensemble (Voting Classifier)
        self.models['Ensemble'] = VotingClassifier(
            estimators=[
                ('svm', self.models['SVM']),
                ('rf', self.models['Random Forest']),
                ('gb', self.models['Gradient Boosting']),
                ('lr', self.models['Logistic Regression'])
            ],
            voting='soft'
        )
        
        print(f"Created {len(self.models)} models.")
        
    def train_models(self):
        """
        Train all models
        """
        print("Training models...")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            print(f"{name} training completed.")
            
    def evaluate_models(self):
        """
        Evaluate all models and calculate performance metrics
        """
        print("Evaluating models...")
        
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            # AUC calculation
            if y_pred_proba is not None:
                auc_score = roc_auc_score(self.y_test, y_pred_proba)
            else:
                auc_score = 0
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'auc_score': auc_score,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'confusion_matrix': confusion_matrix(self.y_test, y_pred)
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc_score:.4f}")
            
    def visualize_results(self):
        """
        Visualize model results
        """
        print("Creating visualizations...")
        
        # Model comparison graphs
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        f1_scores = [self.results[name]['f1_score'] for name in model_names]
        auc_scores = [self.results[name]['auc_score'] for name in model_names]
        
        # Accuracy comparison
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, accuracies, color='skyblue', alpha=0.7)
        plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12)
        plt.xlabel('Models', fontsize=12)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # F1 Score comparison
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, f1_scores, color='lightgreen', alpha=0.7)
        plt.title('Model F1 Score Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('F1 Score', fontsize=12)
        plt.xlabel('Models', fontsize=12)
        plt.ylim(0, 1)
        
        for bar, f1 in zip(bars, f1_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('model_f1_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # AUC comparison
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, auc_scores, color='lightcoral', alpha=0.7)
        plt.title('Model AUC Score Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('AUC Score', fontsize=12)
        plt.xlabel('Models', fontsize=12)
        plt.ylim(0, 1)
        
        for bar, auc in zip(bars, auc_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('model_auc_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Confusion matrices for each model
        for name in model_names:
            cm = self.results[name]['confusion_matrix']
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.label_names, yticklabels=self.label_names)
            plt.title(f'{name} - Confusion Matrix', fontsize=16, fontweight='bold')
            plt.ylabel('Actual', fontsize=12)
            plt.xlabel('Predicted', fontsize=12)
            
            filename = f"{name.lower().replace(' ', '_')}_confusion_matrix.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.show()
            
        print("Visualizations saved.")
        
    def find_best_model(self):
        """
        Find the best performing model based on F1 score
        
        Returns:
            str: Name of the best model
        """
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['f1_score'])
        
        best_f1 = self.results[best_model_name]['f1_score']
        best_accuracy = self.results[best_model_name]['accuracy']
        best_auc = self.results[best_model_name]['auc_score']
        
        print(f"\nBest Model: {best_model_name}")
        print(f"F1 Score: {best_f1:.4f}")
        print(f"Accuracy: {best_accuracy:.4f}")
        print(f"AUC Score: {best_auc:.4f}")
        
        return best_model_name
        
    def save_model(self, model_name, filename):
        """
        Save model and related information
        
        Args:
            model_name (str): Name of the model to save
            filename (str): Filename to save
        """
        import joblib
        
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return
        
        # Save model
        model_filename = f"{filename}_model.pkl"
        joblib.dump(self.models[model_name], model_filename)
        
        # Save model information
        model_info = {
            'label_names': self.label_names,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'selector': self.selector,
            'fruit_type_names': getattr(self, 'fruit_type_names', None),
            'results': self.results.get(model_name, {})
        }
        
        info_filename = f"{filename}_model.pkl_info.pkl"
        joblib.dump(model_info, info_filename)
        
        print(f"Model saved: {model_filename}")
        print(f"Model info saved: {info_filename}")
        
    def predict(self, model_name, image_path):
        """
        Make prediction for a single image
        
        Args:
            model_name (str): Name of the model to use
            image_path (str): Path to the image file
            
        Returns:
            str: Prediction result
        """
        if model_name not in self.models:
            return f"Model {model_name} not found!"
        
        try:
            # Load and process image
            img = io.imread(image_path)
            features = self.extract_features(img)
            
            if features is None:
                return "Could not extract features from image"
            
            # Transform features (scaling and selection)
            features = features.reshape(1, -1)
            if self.scaler:
                features = self.scaler.transform(features)
            if self.selector:
                features = self.selector.transform(features)
            
            # Make prediction
            prediction = self.models[model_name].predict(features)[0]
            prediction_label = self.label_names[prediction]
            
            # Get prediction probability if available
            if hasattr(self.models[model_name], 'predict_proba'):
                proba = self.models[model_name].predict_proba(features)[0]
                confidence = max(proba)
                return f"{prediction_label} (confidence: {confidence:.2f})"
            else:
                return prediction_label
                
        except Exception as e:
            return f"Error during prediction: {str(e)}"

def main():
    """
    Main function to run the fruit ripeness detection system
    """
    # Set data folder path
    data_folder = "dataset/Ripe & Unripe Fruits"
    
    # Check if data folder exists
    if not os.path.exists(data_folder):
        print(f"Error: Data folder not found: {data_folder}")
        print("Please make sure the dataset is downloaded and placed in the correct location.")
        return
    
    # Get selected fruit from command line arguments
    selected_fruit = None
    if len(sys.argv) > 1:
        selected_fruit = sys.argv[1]
        print(f"Selected fruit: {selected_fruit}")
    
    # Create fruit ripeness detection instance
    detector = FruitRipenessDetection(data_folder)
    
    # Load data
    X, y = detector.load_data(selected_fruit)
    
    if len(X) == 0:
        print("No data loaded! Please check the dataset path and structure.")
        return
    
    # Preprocess data
    detector.preprocess_data()
    
    # Feature selection
    detector.feature_selection(k=15)
    
    # Create and train models
    detector.create_models()
    detector.train_models()
    
    # Evaluate models
    detector.evaluate_models()
    
    # Visualize results
    detector.visualize_results()
    
    # Find and save the best model
    best_model = detector.find_best_model()
    detector.save_model(best_model, "best_fruit_ripeness")
    
    print("\nTraining completed successfully!")
    print("You can now use test.py to test the model or UI.py for graphical interface.")

if __name__ == "__main__":
    main()