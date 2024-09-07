# Depth Estimation Using Stereo Images

## Overview

This module implements a depth estimation model using stereo image pairs. By leveraging a custom distance calculation layer, the model accurately estimates object depth from disparity maps, making it ideal for applications in autonomous vehicles, robotics, and 3D reconstruction.

This can be utilised by the drone to navigate complex geometries without risk of collision through stereoscopic imaging.

## Features

- **Custom Distance Layer**: Calculates depth using stereo camera principles.
- **Modular Architecture**: Easy adaptation to various image sizes and resolutions.
- **Synthetic Dataset Support**: Includes scripts to generate and train on synthetic datasets.

## Architecture

1. **Fully Connected Layers**: Feature extraction and disparity prediction from stereo images.
2. **Custom Distance Layer**: Converts disparities to depth estimates based on camera parameters.
3. **Final Output**: Distance map representing pixel depth.

## Dataset

### Synthetic Data Generation

Generate stereo image pairs with configurable parameters like image size and disparity range. Each pair is labeled with ground truth distances for supervised training.

## Results

The model trained on 100 synthetic images achieved:

- **Mean Absolute Error (MAE)**: 0.02 meters
- **Mean Absolute Percentage Error (MAPE)**: 5%

These results highlight the model's potential for accurate depth estimation, with further improvements possible through tuning and larger datasets.