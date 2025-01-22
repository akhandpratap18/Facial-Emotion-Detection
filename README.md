# Facial Emotion Detection

This project implements a **Facial Emotion Detection System** that identifies emotions from facial images. It uses a Convolutional Neural Network (CNN) trained on the [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) from Kaggle to classify emotions into seven categories such as happy, sad, angry, surprised, digust, fear, and neutral.

## Features
- Emotion classification from facial images.
- Saved training CNN model for future integrations.
- Easy-to-use interface for prediction.

## Dataset
The project uses the FER-2013 dataset, a public dataset available on Kaggle. It contains facial images categorized into various emotion labels. It contained about 36000 images in grayscale(single channel).


## Prerequisites
Before running this project, ensure you have the following installed:
- Python 3.8+
- Jupyter Notebook
- Pip (Python package manager)

### Required Libraries
Install the required Python libraries using the following command:
```bash
pip install -r requirements.txt
```

## How to Run the Project
1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/your-username/facial-emotion-detection.git
   cd facial-emotion-detection
   ```

2. Place the FER-2013 dataset files in the `data` directory within the project.

3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. In the Jupyter Notebook interface:
   - Open the `Final.ipynb` notebook to:
     - Load the dataset.
     - Train the model and save it as `emotiondetector1.h5`.
   - Open the `Realtime.ipynb` notebook to:
     - Load the saved model (`emotiondetector1.h5`).
     - Perform real-time emotion detection on input images.

5. To predict emotions from an image in real-time, upload the image to the specified location in the `Realtime.ipynb` notebook and follow the instructions in the cells.

## Project Files
- `Final.ipynb`: Jupyter Notebook containing the training code and saving the model.
- `Realtime.ipynb`: Jupyter Notebook for real-time emotion detection using the saved model.
- `emotiondetector1.h5`: Pre-trained model weights.
- `requirements.txt`: List of dependencies.

## Brief Description of the Project
Facial Emotion Detection is a critical application of computer vision and deep learning, enabling systems to understand and respond to human emotions. This project uses a Convolutional Neural Network (CNN) architecture to classify facial expressions into multiple emotion categories. By leveraging the FER-2013 dataset, the system can achieve an accuracy of approximately **91% on the training set** and **65% on the validation set**. This performance reflects the inherent challenges of the dataset, such as low-resolution images and varied lighting conditions. Despite these challenges, the system provides a solid foundation for applications in human-computer interaction, surveillance, and mental health analysis.

