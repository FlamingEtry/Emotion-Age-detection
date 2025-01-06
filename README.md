# Emotion-Age-detection
This project is a Python-based application designed to detect emotions and estimate ages from images. It leverages pre-trained deep learning models to perform these tasks effectively.

# Emotion and Age Detection Application

This Python application uses deep learning models to detect emotions and estimate ages from images or video streams. It employs pre-trained neural networks to analyze faces and provide predictions.

## Features

- **Real-time Emotion Detection**: Identify emotions such as Angry, Happy, Neutral, Sad, and Surprise from live camera feed or image files.
- **Age Estimation**: Predict age categories (e.g., 0-2, 3-9, 10-19, etc.) from facial images.
- **Image and Video Processing**: Process images from files or analyze video streams using your webcam.
- **Result Saving**: Saves annotated images with detected emotions and ages in a designated output folder.

## Project Structure

```
.idea/                       # IDE configuration files
ages_images/                 # Folder for age estimation-related images
emotions_and_ages_images/    # Combined dataset or testing images
venv/                        # Python virtual environment
Age_model.h5                 # Pre-trained model for age estimation
Emotion_Detection.h5         # Pre-trained model for emotion detection
main.py                      # Main Python script for the application
PVZ.jpg                      # Example input image for testing
```

## Prerequisites

- Python 3.8 or higher
- Required Python libraries:
  - `keras`
  - `opencv-python`
  - `numpy`

### Install Dependencies

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
2. Activate the virtual environment:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
3. Install the required libraries:
   ```bash
   pip install keras opencv-python numpy
   ```

## Usage

1. **Run the application**:
   ```bash
   python main.py
   ```

2. **Choose an option**:
   - Option 1: Process video from your webcam.
   - Option 2: Process an image file.

3. **Input Image**:
   - Provide the path to the image when prompted.

4. **Output**:
   - Detected emotions and ages are displayed on the screen.
   - Annotated images are saved in the `emotions_and_ages_images` folder.

## Example Workflow

1. **Image Processing**:
   - Input: `PVZ.jpg`
   - Output: Annotated image saved as `emotions_and_ages_images/emotion_Happy.jpg`.

2. **Video Processing**:
   - Real-time emotion and age detection using your webcam.

## Models Used

- `Age_model.h5`: A pre-trained age classification model categorizing faces into age groups.
- `Emotion_Detection.h5`: A model for classifying facial expressions into predefined emotion labels.

## Folder Details

- **`emotions_and_ages_images/`**:
  - Contains output images with overlaid predictions.
- **`venv/`**:
  - Python virtual environment for the project dependencies.

## Notes

- Ensure the required Haar cascade XML file (`haarcascade_frontalface_default.xml`) is present in your working directory or update its path in the code.
- Press `q` during webcam processing to quit the application.

## License

This project is open-source and free to use under the [MIT License](LICENSE).
