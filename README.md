# Cryomalus: Apple Orchard Management System

**Cryomalus** is a comprehensive system for monitoring and managing apple orchards using drones and computer vision. It delivers real-time insights into tree health, nutrient levels, pest infestations, and production estimates, combining advanced data collection and analysis to optimize orchard management.

## Key Features

- **Thermal Anomaly Detection**: Utilizes a custom Convolutional Neural Network (CNN) trained in PyTorch to detect thermal anomalies in apple trees, predicting the coordinates of anomalies in thermal images.
- **Nutrient Monitoring**: Tracks and displays nutrient levels such as nitrogen, phosphorus, and potassium, providing detailed insights into soil conditions and tree health.
- **Pest Control**: Monitors and reports pest counts, including aphids, mites, and worms, helping manage pest-related issues effectively.
- **Production Estimation**: Estimates the orchard's yield based on data collected from drone surveys, offering accurate predictions for planning and resource allocation.

## Project Modules

### 1. **Thermal Anomaly Detection with CNN**

- **Files**: `neural_network_files/train_model.py`, `thermal_cnn.py`
- **Technologies**: PyTorch, pandas, numpy
- **Highlights**:
  - **CNN Architecture**: Defines a robust CNN model for thermal anomaly detection.
  - **Training & Inference**: Implements a training script and inference pipeline to detect and predict anomaly locations in thermal images.

### 2. **Dashboard for Real-Time Orchard Monitoring**

- **Files**: `pages/index.js`, `components/Dashboard.js`
- **Technologies**: React, Next.js, Mapbox, TailwindCSS
- **Highlights**:
  - **Map Visualization**: Integrates Mapbox for a geographic representation of the orchard.
  - **Data Tables**: Displays tree health, pest counts, and nutrient levels in a user-friendly interface.
  - **Production Estimates**: Shows current yield projections, aiding in efficient orchard management.

### 3. **Apple-Classifier - Computer Vision Module**

- **Files**: `main.py`, `apple_classifier.py`
- **Technologies**: PyTorch, Transformers, Pillow (PIL), Matplotlib
- **Highlights**:
  - **Early Disease Detection**: Employs machine learning to identify diseases from drone-captured images.
  - **Bounding Box Visualization**: Highlights regions of interest in images for detailed analysis.
  - **Dense Region Captioning**: Provides comprehensive descriptions of specific regions, enhancing understanding and decision-making.

## Other Features

- **Drone Autopilot Integration**: Implement automated flight paths for efficient data collection.
- **Real-Time Notifications**: Develop a notification system for critical issues detected in the orchard.
- **Model Optimization**: Optimize models for faster inference on edge devices, improving overall system efficiency.

Cryomalus represents a significant advancement in orchard management, leveraging cutting-edge technology to deliver actionable insights and enhance decision-making for improved yie
