# Setup Guide  

## 1️⃣ Prerequisites  
Before running this project, ensure you have the following installed:  
- Python 3.x  
- TensorFlow  
- NumPy, Pandas, Scikit-learn  
- FastAPI (for backend)  
- WebSockets (for real-time data streaming)  

## 2️⃣ Dataset Preparation  
We used raw sensor data from the **Smartphone-Based Recognition of Human Activities and Postural Transitions** dataset. The preprocessing steps include:  
- Merging raw sensor data files to create structured training data.  
- Using **SMOTE** to balance the class distribution.  
- Splitting the data into sequences of **50 time steps** for training.  
- Training **RNN, LSTM, and GRU models** for **50 epochs**, using **ReduceLROnPlateau** for better learning.  

## 3️⃣ Installation and Setup  

Navigate to the deployment folder in src using Jupyter Notebook or VS Code. Then, open a terminal inside the project directory and run:  

```
pip install -r requirements.txt
```

## 4️⃣ Running Real-Time Prediction  

### ✅ Install Sensor Server App  
- Download and install the **Sensor Server App** from GitHub:  
  [https://github.com/umer0586/SensorServer](https://github.com/umer0586/SensorServer)  

### ✅ Connect to Mobile Hotspot  
- Ensure your **system is connected to your mobile’s hotspot** for proper communication.  

### ✅ Set IPv4 Address in `main.py`  
1. Find your system’s **IPv4 address** (run `ipconfig` on Windows or `ifconfig` on Linux/macOS).  
2. Update this IP **inside the Sensor Server App** and also **inside `main.py`** under `SENSOR_SERVER_URL`.  
3. Make sure the **WebSocket port number matches** in both places.  

### ✅ Start the FastAPI Backend  

```
python main.py
```

### ✅ Start Streaming Sensor Data  
- Open the Sensor Server App on your phone and start streaming sensor data.  

### ✅ View Predictions  
- Open `frontend/display.html` in a web browser to see real-time predictions.  
- You can also **see live continuous predictions** in the terminal while `main.py` is running.  

---
This guide ensures a smooth setup for training and running real-time predictions. 🚀  
