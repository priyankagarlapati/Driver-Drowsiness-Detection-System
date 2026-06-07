# Driver Drowsiness Detection System

## Overview

The Driver Drowsiness Detection System is an AI-powered safety application that monitors a driver's eye movements in real time and detects signs of drowsiness. The system uses Computer Vision and Deep Learning techniques to identify eye states (open or closed) and generates an alert when signs of fatigue are detected, helping reduce the risk of road accidents.

## Features

* Real-time driver monitoring using a webcam
* Eye detection using Haar Cascade Classifiers
* Deep Learning-based eye state classification
* Audio alert generation for drowsiness detection
* Flask-based web interface
* Real-time fatigue monitoring and safety assistance

## Technologies Used

* Python
* OpenCV
* TensorFlow / Keras
* Flask
* NumPy
* HTML
* CSS
* JavaScript

## Project Structure

```text
Driver-Drowsiness-Detection-System/
│
├── app.py
├── model.py
├── alarm.wav
├── models/
├── data/
├── static/
├── templates/
└── haar cascade files/
```

## How It Works

1. Captures live video through the webcam.
2. Detects the driver's face and eyes using Haar Cascade Classifiers.
3. Classifies eye states as Open or Closed using a trained CNN model.
4. Continuously monitors eye closure duration.
5. Triggers an audio alert when drowsiness is detected.

## Installation

```bash
git clone https://github.com/priyankagarlapati/Driver-Drowsiness-Detection-System.git
cd Driver-Drowsiness-Detection-System
python app.py
```

## Applications

* Driver Safety Systems
* Smart Transportation Solutions
* Fleet Monitoring Systems
* Automotive Safety Research

## Future Enhancements

* Head pose estimation
* Mobile application integration
* Advanced fatigue analytics
* Cloud-based monitoring dashboard

## Author

Priyanka Garlapati
