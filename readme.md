# Breast Cancer Analyzer (IDC Detection)
This project implements a Deep Learning solution for detecting Invasive Ductal Carcinoma (IDC) in histopathological images. It utilizes a Convolutional Neural Network (CNN) built with TensorFlow and Keras to classify images as either IDC positive or negative. The project includes a graphical user interface (GUI) for easy model inference on new images.

## Project Overview
The system performs the following operations:

1. Data Loading & Preprocessing: Loads images from a structured directory, resizes them to 50x50 pixels, and normalizes pixel values.
2. Model Training: Trains a custom CNN architecture with data augmentation to handle variability and prevent overfitting.

3. Evaluation: Calculates accuracy and loss metrics on a dedicated test set.

4. Inference: Provides a Tkinter-based desktop application for real-time classification of user-selected images.

## Technologies Used
- Language: Python 3.x
- Deep Learning: TensorFlow, Keras
- Computer Vision: OpenCV, NumPy
- Data processing: Scikit-learn (Train/Test split), Tqdm
- Visualization: Matplotlib
- GUI: Tkinter

## Dataset Structure
The project expects the dataset to be located in the root directory. The folder structure should be organized by patient IDs, containing subfolders 0 (non-IDC) and 1 (IDC).
```
project_root/
│
├── DC_regular_ps50_idx5/      # Main Dataset Directory
│   ├── 10253/                 # Patient ID
│   │   ├── 0/                 # Negative samples
│   │   └── 1/                 # Positive samples
│   └── ...
│
├── model_breast_cancer_cnn.h5 # Saved model (generated after training)
└── main.py                    # Main application script
```

## Installation
To run this project, ensure you have Python installed along with the required dependencies.

```
pip install tensorflow opencv-python numpy matplotlib scikit-learn tqdm
```
## Usage
Ensure the dataset folder (DC_regular_ps50_idx5) is present in the project directory.

Run the main script:

```
python main.py
```
Upon execution, the script will:

1. Load and process the image data.

2. Train the CNN model (displaying progress bars).

3. Output the test accuracy and loss.

4. Display training graphs (Accuracy/Loss over epochs).

5. Save the model as model_breast_cancer_cnn.h5.

6. Launch the GUI window for manual testing.

## Model Architecture
The Convolutional Neural Network consists of the following layers:

* **Input Layer:** 50x50x3 (RGB images)
* **Convolutional Blocks:** 4 blocks, each containing:
  * Conv2D (Filters: 32, 64, 128, 256)
  * ReLU Activation
  * MaxPooling2D (2x2)
* **Fully Connected Layers:**
  * Flatten
  * Dense (256 units, ReLU)
  * Dropout (0.5) for regularization
* **Output Layer:** Dense (1 unit, Sigmoid activation)

## Performance
The model was evaluated on a held-out test set with the following results:

* Test Accuracy: ~85.5%

* Test Loss: ~0.32