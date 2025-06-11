"""
Fruit Ripeness Detection - Simple and Reliable Interface
"""

import os
import sys
import joblib
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import traceback
from datetime import datetime

# Import main module
from app import FruitRipenessDetection

class FruitRipenessUI:
    def __init__(self, master):
        # Main window configuration
        self.master = master
        self.master.title("Fruit Ripeness Detection")
        self.master.geometry("900x650")
        self.master.configure(bg="#f5f5f5")
        
        # Required variables
        self.model = None
        self.model_name = None
        self.detector = None
        self.selected_image_path = None
        self.status_text = tk.StringVar()
        self.status_text.set("Ready - Please load an image")
        
        # Log file
        self.log_file = open("ui_debug.log", "a")
        self.log_file.write(f"\n\n=== NEW SESSION: {datetime.now()} ===\n")
        
        # Create UI components
        self.create_ui()
        
        # Load model
        self.load_model()
        
    def log(self, message):
        """Write message to log file"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_file.write(f"[{timestamp}] {message}\n")
        self.log_file.flush()  # Write to disk immediately
        
    def create_ui(self):
        """Create user interface"""
        self.log("Creating interface...")
        
        # Main frame
        main_frame = tk.Frame(self.master, bg="#f5f5f5")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        header_frame = tk.Frame(main_frame, bg="#f5f5f5")
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = tk.Label(
            header_frame, 
            text="Fruit Ripeness Detection",
            font=("Arial", 22, "bold"),
            bg="#f5f5f5",
            fg="#333333"
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            header_frame,
            text="Load a fruit image and check its ripeness status",
            font=("Arial", 12),
            bg="#f5f5f5",
            fg="#555555"
        )
        subtitle_label.pack(pady=(5, 0))
        
        # Content section - two column layout
        content_frame = tk.Frame(main_frame, bg="#f5f5f5")
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Image
        left_frame = tk.LabelFrame(content_frame, text="Fruit Image", font=("Arial", 12, "bold"), bg="#f5f5f5")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.image_container = tk.Frame(left_frame, bg="white", width=400, height=400)
        self.image_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.image_container.pack_propagate(False)
        
        self.image_label = tk.Label(
            self.image_container,
            text="Please click 'Select Image' button",
            font=("Arial", 12),
            bg="white",
            fg="#888888"
        )
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Right panel - Results
        right_frame = tk.Frame(content_frame, bg="#f5f5f5")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Result panel
        result_frame = tk.LabelFrame(right_frame, text="Ripeness Prediction", font=("Arial", 12, "bold"), bg="#f5f5f5")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.result_label = tk.Label(
            result_frame,
            text="Please load an image",
            font=("Arial", 14, "bold"),
            bg="white",
            fg="#333333",
            height=5
        )
        self.result_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Features panel
        features_frame = tk.LabelFrame(right_frame, text="Important Features", font=("Arial", 12, "bold"), bg="#f5f5f5")
        features_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview frame
        tree_frame = tk.Frame(features_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Treeview style
        style = ttk.Style()
        style.configure("Treeview", font=("Arial", 11))
        style.configure("Treeview.Heading", font=("Arial", 11, "bold"))
        
        # Treeview
        self.feature_tree = ttk.Treeview(
            tree_frame,
            columns=("Feature", "Value"),
            show="headings",
            height=8,
            yscrollcommand=scrollbar.set
        )
        self.feature_tree.heading("Feature", text="Feature")
        self.feature_tree.heading("Value", text="Value")
        self.feature_tree.column("Feature", width=150)
        self.feature_tree.column("Value", width=150)
        self.feature_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=self.feature_tree.yview)
        
        # Buttons
        button_frame = tk.Frame(main_frame, bg="#f5f5f5")
        button_frame.pack(fill=tk.X, pady=20)
        
        # Select Image button
        self.select_button = tk.Button(
            button_frame,
            text="Select Image",
            command=self.browse_image,
            font=("Arial", 12, "bold"),
            bg="#2196F3",
            fg="white",
            padx=20,
            pady=10,
            relief=tk.RAISED,
            bd=2
        )
        self.select_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Analyze button
        self.analyze_button = tk.Button(
            button_frame,
            text="Analyze Ripeness",
            command=self.analyze_image,
            font=("Arial", 12, "bold"),
            bg="#FF9800",
            fg="white",
            padx=20,
            pady=10,
            relief=tk.RAISED,
            bd=2,
            state=tk.DISABLED
        )
        self.analyze_button.pack(side=tk.LEFT)
        
        # Status bar
        status_bar = tk.Label(
            self.master,
            textvariable=self.status_text,
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            font=("Arial", 10),
            bg="#f0f0f0",
            padx=10,
            pady=5
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def load_model(self):
        """Load trained model"""
        self.log("Loading model...")
        self.status_text.set("Loading model...")
        
        try:
            # Find model files
            model_files = [f for f in os.listdir('.') if f.endswith('_model.pkl') and not f.endswith('_info.pkl')]
            
            if not model_files:
                self.log("No model found")
                self.status_text.set("Error: No trained model found")
                messagebox.showerror("Error", 
                    "No trained model found!\n\n"
                    "Please train a model first by running:\n"
                    "python app.py")
                return
            
            # Use the first model found
            model_path = model_files[0]
            info_path = f"{model_path}_info.pkl"
            
            self.log(f"Loading model: {model_path}")
            
            # Load model
            self.model = joblib.load(model_path)
            
            # Load model information
            if os.path.exists(info_path):
                model_info = joblib.load(info_path)
                self.log("Model information loaded")
            else:
                model_info = {}
                self.log("No model information file found")
            
            # Create detector instance
            data_folder = "dataset/Ripe & Unripe Fruits"
            self.detector = FruitRipenessDetection(data_folder)
            
            # Transfer model information
            self.detector.label_names = model_info.get('label_names', np.array(['unripe', 'ripe']))
            self.detector.feature_names = model_info.get('feature_names', [])
            self.detector.scaler = model_info.get('scaler', None)
            self.detector.selector = model_info.get('selector', None)
            
            # Set model name
            self.model_name = os.path.basename(model_path).replace('_model.pkl', '').replace('_', ' ').title()
            self.detector.models[self.model_name] = self.model
            
            self.log(f"Model loaded successfully: {self.model_name}")
            self.status_text.set(f"Model loaded: {self.model_name}")
            
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            self.log(error_msg)
            self.status_text.set("Error loading model")
            messagebox.showerror("Error", error_msg)
            
    def browse_image(self):
        """Browse and select image file"""
        self.log("Image selection dialog opened")
        
        # File dialog
        file_path = filedialog.askopenfilename(
            title="Select Fruit Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.log(f"Image selected: {file_path}")
            self.selected_image_path = file_path
            self.process_image(file_path)
        else:
            self.log("Image selection cancelled")
            
    def process_image(self, file_path):
        """Process and display selected image"""
        try:
            self.log(f"Processing image: {file_path}")
            self.status_text.set("Processing image...")
            
            # Load and resize image for display
            pil_image = Image.open(file_path)
            
            # Calculate aspect ratio preserving resize
            display_size = (380, 380)
            pil_image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(pil_image)
            
            # Update image label
            self.image_label.configure(image=self.photo, text="")
            
            # Enable analyze button
            self.analyze_button.config(state=tk.NORMAL)
            
            # Clear previous results
            self.result_label.config(text="Image loaded\nClick 'Analyze Ripeness' to proceed")
            
            # Clear feature tree
            for item in self.feature_tree.get_children():
                self.feature_tree.delete(item)
            
            self.log("Image processing completed")
            self.status_text.set("Image loaded - Ready for analysis")
            
        except Exception as e:
            error_msg = f"Error processing image: {str(e)}"
            self.log(error_msg)
            self.status_text.set("Error processing image")
            messagebox.showerror("Error", error_msg)
            
    def analyze_image(self):
        """Analyze image ripeness"""
        if not self.selected_image_path or not self.model:
            self.log("Analysis attempted without image or model")
            messagebox.showerror("Error", "Please select an image and ensure model is loaded")
            return
        
        try:
            self.log(f"Starting analysis for: {self.selected_image_path}")
            self.status_text.set("Analyzing ripeness...")
            
            # Make prediction
            prediction = self.detector.predict(self.model_name, self.selected_image_path)
            
            # Extract confidence if available
            if "confidence:" in prediction:
                parts = prediction.split("confidence:")
                pred_label = parts[0].strip().rstrip("(").strip()
                confidence = parts[1].strip().rstrip(")").strip()
                
                display_text = f"Prediction: {pred_label.upper()}\n\nConfidence: {confidence}"
                
                # Set color based on prediction
                if pred_label.lower() == "ripe":
                    color = "#4CAF50"  # Green
                else:
                    color = "#FF9800"  # Orange
            else:
                display_text = f"Prediction: {prediction.upper()}"
                color = "#333333"
            
            # Update result display
            self.result_label.config(text=display_text, fg=color)
            
            # Extract and display features
            self.extract_and_display_features()
            
            self.log(f"Analysis completed: {prediction}")
            self.status_text.set("Analysis completed")
            
        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}"
            self.log(error_msg)
            self.status_text.set("Error during analysis")
            messagebox.showerror("Error", error_msg)
            
    def extract_and_display_features(self):
        """Extract features from image and display them"""
        try:
            if not self.detector:
                return
            
            # Load image
            from skimage import io
            img = io.imread(self.selected_image_path)
            
            # Extract features
            features = self.detector.extract_features(img)
            
            if features is not None and self.detector.feature_names:
                # Clear previous entries
                for item in self.feature_tree.get_children():
                    self.feature_tree.delete(item)
                
                # Add features to tree
                for i, (name, value) in enumerate(zip(self.detector.feature_names, features)):
                    # Format value for display
                    if isinstance(value, (int, float)):
                        formatted_value = f"{value:.3f}"
                    else:
                        formatted_value = str(value)
                    
                    # Insert into tree
                    self.feature_tree.insert("", "end", values=(name, formatted_value))
                
                self.log(f"Displayed {len(features)} features")
            
        except Exception as e:
            self.log(f"Error extracting features: {str(e)}")
            
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'log_file'):
            self.log_file.close()

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = FruitRipenessUI(root)
    
    # Handle window closing
    def on_closing():
        app.log("Application closing")
        if hasattr(app, 'log_file'):
            app.log_file.close()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start main loop
    root.mainloop()

if __name__ == "__main__":
    main()