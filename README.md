COVID-19 Chest X-ray Detection Project
=====================================

Overview
--------
This project builds and serves a binary chest X-ray classifier that predicts:
1. COVID Positive
2. Normal

The codebase provides:
- A training pipeline using TensorFlow/Keras and MobileNetV2 transfer learning
- A single-image prediction script for quick CLI testing
- A desktop GUI application (Tkinter) for interactive inference
- Shared preprocessing and model-loading utilities

Important: This software is for educational and research use only. It is not a certified medical device and must not be used as a standalone diagnostic system.


Scope of This README
--------------------
This README documents only the coding part of the repository (the project folder and root project setup), as requested.
It does not analyze or document dataset internals from:
- COVID-19_Radiography_Dataset/
- COVID-19_Test_Data/


Repository Coding Structure
---------------------------
Root
- LICENSE
- README.txt (this file)
- project/

project/
- main.py
  - Main entry point for launching the GUI app
- train_model.py
  - CLI script to train and save model as covid_model.h5
- test_model.py
  - CLI script to run prediction on one image
- covid_model.h5
  - Saved trained model used by GUI and test script
- requirements.txt
  - Python dependencies: numpy, Pillow, tensorflow
- gui/
  - app.py: Tkinter GUI logic and UX workflow
- model/
  - train.py: dataset loading, model definition, training, checkpointing
  - predict.py: inference and output interpretation
  - load_model.py: robust model loader with backend fallback
- utils/
  - preprocess.py: image preprocessing pipeline
- assets/
  - placeholder folder currently containing .gitkeep


How the Application Works
-------------------------
1. User launches the GUI (main.py).
2. GUI tries to load project/covid_model.h5 using model/load_model.py.
3. User uploads an X-ray image.
4. Image is preprocessed to shape (1, 224, 224, 3) and normalized to [0, 1].
5. Model predicts.
6. Prediction is interpreted as:
   - sigmoid output (1 neuron): >= 0.5 => Normal, otherwise COVID Positive
   - softmax output (>= 2 neurons): index mapping [COVID Positive, Normal]
7. GUI displays class label and confidence.


Detailed Module Explanation
---------------------------
1) GUI Layer
------------
File: project/gui/app.py

Key responsibilities:
- Builds complete desktop UI with Tkinter
- Handles image upload and preview
- Runs inference in a background thread to keep UI responsive
- Displays prediction result and confidence
- Manages button state, loading bar, reset behavior, and error dialogs

UI behavior notes:
- If model cannot be loaded, app still starts but shows status warning
- Predict action validates image selection before inference
- Result color coding:
  - Red-ish for COVID Positive
  - Green for Normal


2) Model Loading
----------------
File: project/model/load_model.py

Key responsibilities:
- Resolves and imports load_model dynamically
- Tries tensorflow.keras.models first
- Falls back to keras.models if needed
- Caches loaded model using lru_cache(maxsize=1)
- Returns None instead of crashing when model/backend is unavailable

Why this design matters:
- Improves compatibility across TensorFlow/Keras environments
- Prevents app crash during import when backend is missing


3) Inference Logic
------------------
File: project/model/predict.py

Key responsibilities:
- Validates model availability
- Calls preprocess_image from utils/preprocess.py
- Runs model.predict(..., verbose=0)
- Supports both binary sigmoid and multiclass-style softmax outputs
- Returns tuple: (label, confidence)

Current thresholding:
- NORMAL_THRESHOLD = 0.50 for sigmoid output models


4) Preprocessing
----------------
File: project/utils/preprocess.py

Pipeline:
- Load image with Pillow
- Convert to RGB
- Resize to 224x224
- Convert to float32 numpy array
- Normalize by 255.0
- Add batch dimension

Returned shape:
- (1, 224, 224, 3)


5) Training Pipeline
--------------------
File: project/model/train.py

Dataset handling:
- Expects exactly these class folders under provided dataset root:
  - COVID/images
  - Normal/images
- Collects only image files with extensions:
  .png, .jpg, .jpeg, .bmp, .tif, .tiff
- Labels:
  - COVID -> 0
  - Normal -> 1
- Splits data into train/validation using shuffled indices

Data pipeline:
- tf.data.Dataset with decoding, resizing, normalization
- Shuffle, batch, and prefetch with AUTOTUNE

Model architecture:
- Base: MobileNetV2 (ImageNet weights, include_top=False, frozen)
- Input augmentation:
  - RandomFlip(horizontal)
  - RandomRotation(0.04)
  - RandomZoom(0.08)
- Head:
  - GlobalAveragePooling2D
  - Dropout(0.25)
  - Dense(1, sigmoid)

Compile settings:
- Optimizer: Adam(1e-3)
- Loss: binary_crossentropy
- Metrics: accuracy, AUC

Training callbacks:
- EarlyStopping on val_accuracy, patience=3, restore_best_weights=True
- ModelCheckpoint saving best model to output path


6) Script Entry Points
----------------------
File: project/train_model.py
- CLI wrapper around train_and_save_model
- Arguments:
  - --data-dir (required)
  - --epochs (default: 6)
  - --batch-size (default: 32)
  - --image-size (default: 224)
- Saves model to project/covid_model.h5

File: project/test_model.py
- CLI prediction for single image path argument
- Loads project/covid_model.h5
- Prints label and confidence

File: project/main.py
- Launches Tkinter application class CovidDetectionApp


Setup and Installation
----------------------
Requirements:
- Python 3.10+ recommended
- pip
- Virtual environment (recommended)

Install dependencies:
1. Open terminal in repository root
2. Run:

Windows PowerShell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r project/requirements.txt


How to Run
----------
1) Launch GUI
-------------
From repository root:
python project/main.py

2) Train a model
----------------
From repository root:
python project/train_model.py --data-dir "COVID-19_Radiography_Dataset" --epochs 6 --batch-size 32 --image-size 224

Notes:
- Training code uses only COVID/images and Normal/images under the provided data dir
- Trained model is saved to project/covid_model.h5

3) Test one image from CLI
--------------------------
From repository root:
python project/test_model.py "path/to/xray_image.png"


Expected Data Folder Format for Training
----------------------------------------
The path passed to --data-dir must contain:
- COVID/images/
- Normal/images/

Example:
COVID-19_Radiography_Dataset/
- COVID/images/...
- Normal/images/...

Other class folders in the dataset are ignored by current training code.


Current Limitations
-------------------
- Training uses only two classes (COVID and Normal)
- No segmentation mask usage in training pipeline
- No explicit class balancing, sampling strategy, or weighted loss
- No evaluation script for confusion matrix, ROC export, or detailed metrics report
- Model input size fixed to 224x224 in preprocess path


Troubleshooting
---------------
1. GUI shows model not loaded
- Ensure project/covid_model.h5 exists and is a valid Keras/TensorFlow model

2. Import/backend errors
- Reinstall dependencies from project/requirements.txt
- Ensure TensorFlow installation matches your Python version

3. Invalid image error
- Confirm file is a readable image format supported by Pillow

4. Training fails due to folder structure
- Verify data dir includes both COVID/images and Normal/images with image files


Suggested Future Improvements
-----------------------------
- Add full evaluation pipeline (precision/recall/F1/confusion matrix)
- Add optional fine-tuning (unfreeze top MobileNetV2 layers)
- Add class imbalance handling and threshold tuning
- Add model/version metadata logging
- Add unit tests for preprocess and prediction output mapping
- Add Docker support and reproducible environment lock file


License
-------
See LICENSE in repository root.
