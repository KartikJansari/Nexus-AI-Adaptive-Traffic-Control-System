Project Overview

This project presents an AI-powered Adaptive Traffic Signal Control System that dynamically allocates green signal time based on real-time traffic density detected from video footage.

Unlike traditional traffic systems that use fixed-time signals, this system uses Computer Vision and Machine Learning to intelligently control traffic flow at a four-way intersection.

The system:

Detects vehicles from 4 different road video feeds

Counts traffic density in each lane

Predicts optimal green signal duration using a trained ML model

Simulates a live traffic signal system with animated countdown

🎯 Problem Statement

Traditional traffic signal systems:

Use fixed signal timings

Do not adapt to real-time congestion

Cause unnecessary waiting time

Increase fuel consumption and pollution

This project solves that by:

Dynamically adjusting signal timings based on real-time vehicle density.

🧠 Proposed Solution

The system performs the following steps:

📤 Upload 4 road video clips (Lane 1–4)

🚗 Detect vehicles using YOLOv8 object detection

📊 Count vehicles in each lane

🤖 Use a trained Random Forest model to predict optimal green signal duration

🚦 Simulate a live 4-lane traffic signal system

⏳ Allocate longer green time to lanes with higher vehicle count

🏗 System Architecture
4 Lane Videos
        ↓
YOLOv8 Vehicle Detection
        ↓
Vehicle Count Per Lane
        ↓
Random Forest ML Model
        ↓
Green Time Prediction
        ↓
Animated Signal Simulation

🔬 Technologies Used
Component	Technology
Programming Language	Python
Computer Vision	YOLOv8 (Ultralytics)
Video Processing	OpenCV
Machine Learning	Scikit-learn (Random Forest Regressor)
Model Saving	Joblib
Frontend Interface	Streamlit
Visualization	HTML + CSS in Streamlit
🤖 Machine Learning Model
Model Used:

Random Forest Regressor

Input Features:

Vehicle count in Lane 1

Vehicle count in Lane 2

Vehicle count in Lane 3

Vehicle count in Lane 4

Output:

Predicted green signal duration for each lane

The model allocates green time proportionally based on real-time traffic density.

🚦 Key Features

✔ 4-Lane Video Upload
✔ Real-Time Vehicle Detection
✔ Intelligent Signal Timing Prediction
✔ Live 4-Lane Signal Simulation
✔ Countdown Animation
✔ Clean Interactive Dashboard
✔ CPU Compatible
✔ Smart City Application

🌍 Sustainable Development Goals (SDG)

This project supports:

SDG 11 – Sustainable Cities and Communities

SDG 13 – Climate Action

By:

Reducing traffic congestion

Minimizing fuel wastage

Lowering carbon emissions

📂 Project Structure
traffic_project/
│
├── models/
│   └── signal_model.pkl
│
├── app.py
├── traffic.ipynb
├── requirements.txt
└── README.md

⚙️ How to Run the Project
1️⃣ Clone the repository
git clone https://github.com/your-username/traffic-ai-system.git
cd traffic-ai-system

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit app
streamlit run app.py

📊 Output Example

Vehicle count per lane displayed

AI predicted green signal duration

Live animated 4-lane signal simulation

Highest traffic lane receives longest green time

🚀 Future Improvements

Add yellow signal transition

Real-time CCTV camera integration

Deploy on cloud (Render / HuggingFace)

Add traffic heatmap analytics

Integrate emergency vehicle priority system

Continuous signal cycle automation

🎓 Learning Outcomes

Through this project, the following concepts were implemented:

Computer Vision using YOLOv8

Object Detection and Counting

Data Simulation for ML Training

Regression Modeling

Real-time Web App Deployment

Smart Infrastructure Simulation

🏆 Project Impact

This project demonstrates how Artificial Intelligence can be used to build:

Smart, adaptive, and efficient urban traffic systems.

It combines:

Deep Learning

Machine Learning

Real-Time Simulation

Web Deployment

into a complete end-to-end AI solution.

👨‍💻 Author

Akash
AI/ML Enthusiast
Smart City & Sustainable Tech Developer
