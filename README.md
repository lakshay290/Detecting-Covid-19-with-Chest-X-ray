# 🦠 COVID-19 Chest X-ray Detection using Deep Learning

🚀 A Computer Vision project that detects COVID-19 from Chest X-ray images using Deep Learning (TensorFlow + MobileNetV2)

--------------------------------------------------

📌 Overview

This project is a binary image classification system that predicts:

🟥 COVID Positive  
🟩 Normal  

It includes:
🧠 Deep Learning model (MobileNetV2 Transfer Learning)  
🖥️ Desktop GUI (Tkinter)  
⚡ CLI prediction script  
🔄 Full training pipeline  

⚠️ Disclaimer  
This project is for educational purposes only and should NOT be used for real medical diagnosis.

--------------------------------------------------

🎯 Features

📷 Upload X-ray images via GUI  
⚡ Fast prediction with confidence score  
🧵 Background processing (no UI freeze)  
🧪 CLI-based testing support  
🔁 Train your own model easily  

--------------------------------------------------

🗂️ Project Structure

computer-vision-project/

project/
│   main.py
│   train_model.py
│   test_model.py
│   covid_model.h5
│   requirements.txt

│   gui/
│       app.py

│   model/
│       train.py
│       predict.py
│       load_model.py

│   utils/
│       preprocess.py

│   assets/

README.md
.gitignore

--------------------------------------------------

🧠 How It Works

1. User uploads an X-ray image  
2. Image is resized to 224x224 and normalized  
3. Model predicts using MobileNetV2  
4. Output:
   🟥 COVID Positive  
   🟩 Normal  
5. Confidence score is displayed  

--------------------------------------------------

📊 Model Details

Architecture: MobileNetV2  
Input Size: 224 x 224 x 3  
Output: Binary Classification  
Optimizer: Adam  
Loss: Binary Crossentropy  

--------------------------------------------------

📥 Dataset

⚠️ Dataset is NOT included due to large size

Download from:
https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

Expected Structure:

COVID-19_Radiography_Dataset/
COVID/images/
Normal/images/

--------------------------------------------------

⚙️ Installation

git clone https://github.com/lakshay290/Detecting-Covid-19-with-Chest-X-ray  


Create virtual environment:
python -m venv .venv  

Activate:
.\.venv\Scripts\activate  

Install dependencies:
pip install -r project/requirements.txt  

--------------------------------------------------

▶️ How to Run

🖥️ Run GUI:
python project/main.py  

🧪 Test Image:
python project/test_model.py "image_path.jpg"  

🧠 Train Model:
python project/train_model.py --data-dir "COVID-19_Radiography_Dataset"  

--------------------------------------------------

📈 Output Example

🟥 COVID Positive (Confidence: 92%)  
🟩 Normal (Confidence: 88%)  

--------------------------------------------------

⚠️ Limitations

Only 2 classes (COVID & Normal)  
No advanced evaluation metrics  
No class balancing  
Fixed input size  

--------------------------------------------------

🚀 Future Improvements

📊 Add evaluation metrics  
⚖️ Handle class imbalance  
🔍 Fine-tune model  
🐳 Add Docker  
🌐 Deploy on web  

--------------------------------------------------

🛠️ Tech Stack

Python 🐍  
TensorFlow / Keras 🤖  
OpenCV / Pillow 📷  
Tkinter 🖥️  

--------------------------------------------------

👨‍💻 Author

Lakshay Bishnoi  

--------------------------------------------------

⭐ Support

Star ⭐ the repo  
Fork 🍴 the project  
Share 📢 with others  

--------------------------------------------------