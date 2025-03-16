# TinyML-Based Human Activity Recognition  

## 📌 Project Overview  
This project focuses on real-time **Human Activity Recognition (HAR)** using sensor data from mobile devices. We trained **RNN, LSTM, and GRU** models to classify human activities using accelerometer and gyroscope readings. The models were optimized for **TinyML deployment**, converted to **TensorFlow Lite (TFLite)**, and tested with real-time sensor data.  

## 🚀 Features  
✔️ **Real-time activity recognition** using mobile sensor data  
✔️ **Optimized deep learning models** (RNN, LSTM, GRU)  
✔️ **SMOTE balancing** to handle class imbalance  
✔️ **Quantized models** for efficient TinyML deployment  
✔️ **FastAPI backend** for handling real-time predictions  
✔️ **Web-based frontend** for live activity display  

## 📂 Project Structure  
```
/src        - Python scripts for training and optimization of models & local deployment
/models     - Pre-trained and optimized models  
/datasets   - Raw and processed sensor data  
/reports    - Detailed Scientific report with results and analysis  
/docs       - Setup and deployment documentation & model comparison graphs 
README.md   - Project overview and instructions  
```

## 🔧 demonstration Setup Instructions  
1️⃣ **Install dependencies:**  
```
pip install -r requirements.txt
```
2️⃣ **Set up the Sensor Server App** ([Download here](https://github.com/umer0586/SensorServer))  
3️⃣ **Connect system to mobile hotspot**  
4️⃣ **Update IPv4 address in `main.py` and Sensor Server App**  
5️⃣ **Run FastAPI backend:**  
```
python main.py
```
6️⃣ **Start sensor streaming from the mobile app**  
7️⃣ **Open `frontend/display.html` to view predictions**  

## 📊 Quantized Model Performance  
| Model | Accuracy | Inference Time (ms) | Model Size (KB) |
|--------|----------|------------------|--------------|
| RNN  | 94.46%  | 2.027  | 162  |
| LSTM | 98.35%  | 2.428  | 185  |
| GRU  | 98.50%  | 3.525  | 152  |

## 📈 Results and Insights  
- **LSTM and GRU** outperform RNN in accuracy due to better temporal dependency handling.  
- **GRU** provides the best balance between accuracy and inference speed.  
- **Quantization significantly reduces model size** with minimal accuracy loss.  
- Real-time tests confirmed smooth predictions with **mobile sensor data streaming**.  

## 🔮 Future Work  
- **Deploy the system online** using **Render for the backend** and **Streamlit for the frontend**.  
- **Apply Butterworth filtering** to real-time sensor data to remove noise and gravity components.  
- **Optimize communication** between the frontend and backend for better latency handling.  

## 📝 Acknowledgments  
- **Dataset:** [Smartphone-Based Human Activity Recognition Dataset](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)  
- **Sensor Streaming:** [Sensor Server App](https://github.com/umer0586/SensorServer) (GPL-3.0 Licensed)  
- **Deep Learning Frameworks:** TensorFlow, FastAPI  

---
This project demonstrates the feasibility of deploying **TinyML-powered real-time HAR systems** for mobile applications. 🚀  
