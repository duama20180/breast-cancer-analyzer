import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import os

try:
    from tensorflow.keras.models import load_model

    print("Keras import successful for GUI")
except ImportError as e:
    print("Failed to import Keras for GUI:", e)
    exit(1)

root = tk.Tk()
root.title("Breast Cancer Classification")
root.geometry("400x250")
root.resizable(False, False)

model_path = 'model_breast_cancer_cnn.h5'

try:
    if not os.path.exists(model_path):
        model_path = 'breast_cancer_cnn.h5'

    model = load_model(model_path)
    print(f"Model loaded from {model_path}")
except Exception as e:
    print("Failed to load model:", e)
    messagebox.showerror("Error", f"Failed to load model: {e}")
    exit(1)


def process_image(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.resize(img, (50, 50))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def predict_image(image_path):
    result_label.config(text="Processing...", fg="black")
    root.update()

    img = process_image(image_path)
    if img is None:
        messagebox.showerror("Error", "Unable to load or process image!")
        result_label.config(text="Error loading image", fg="black")
        return

    prediction = float(model.predict(img)[0][0])

    if prediction > 0.5:
        result_text = "IDC Positive (Cancer Detected)"
        color = "red"
        confidence = prediction
    else:
        result_text = "IDC Negative (Healthy)"
        color = "green"
        confidence = 1 - prediction

    result_label.config(
        text=f"{result_text}\nConfidence: {confidence:.2%}",
        fg=color
    )


def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
    if file_path:
        predict_image(file_path)


def clear_results():
    result_label.config(text="Result: None", fg="black")



open_button = tk.Button(root, text="Select Image Analysis", command=open_file, font=("Arial", 12), height=2, width=20)
open_button.pack(pady=20)

result_label = tk.Label(root, text="Result: None", font=("Arial", 16, "bold"))
result_label.pack(pady=20)

clear_button = tk.Button(root, text="Clear Result", command=clear_results, font=("Arial", 10))
clear_button.pack(pady=10)

root.mainloop()