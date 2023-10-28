## Project Overview

This project focuses on developing an Artificial Intelligence system for lip reading. Lip reading is the process of understanding speech by visually interpreting the movements of the lips, tongue, and facial expressions. The project uses deep learning techniques, specifically Convolutional Neural Networks (CNNs) and Long Short-Term Memory networks (LSTMs), to recognize spoken words from videos of lip movements. The model is trained on a dataset containing videos of spoken phrases and corresponding text transcriptions.

## Files and Directories

- **AI_Project.ipynb**: Jupyter Notebook containing the code for the lip reading model, data loading functions, data preprocessing, and training.
- **data.zip**: Dataset containing video files and corresponding alignment files for training and testing.
- **checkpoints.zip**: Pre-trained model weights stored in checkpoints for making predictions.
- **Readme.md**: Project readme file explaining the project overview, file structure, and usage instructions.

## Steps to Run the Code

### 1. **Data Loading and Preprocessing**:
   - The `data.zip` file contains the dataset. Unzip the file to access the video files and alignment files.
   - Videos are loaded, preprocessed, and transformed into sequences of frames for model input.
   - Text transcriptions are processed and encoded into numerical sequences for model output.

### 2. **Model Architecture**:
   - The neural network architecture consists of Conv3D layers for spatial feature extraction and Bidirectional LSTMs for temporal modeling.
   - The model is trained to predict the sequence of characters (text transcription) corresponding to the lip movements in the input video.

### 3. **Training**:
   - The model is trained using the processed data.
   - Custom CTC loss (Connectionist Temporal Classification) is used to calculate the loss during training.
   - Training progress and examples of predictions are displayed and saved in checkpoints.

### 4. **Prediction**:
   - Pre-trained model weights are loaded from `checkpoints.zip`.
   - The model makes predictions on a sample video and outputs the recognized text transcription.

## Requirements

- Python 3.x
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- gdown

## Usage

1. Clone the repository: `git clone <repository-url>`
2. Navigate to the project directory: `cd <repository-folder>`
3. Ensure all required libraries are installed using: `pip install -r requirements.txt`
4. Run `AI_Project.ipynb` in a Jupyter Notebook environment for data preprocessing, model training, and evaluation.
