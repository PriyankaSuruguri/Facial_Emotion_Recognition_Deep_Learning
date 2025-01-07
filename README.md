# Facial_Emotion_Recognition_Deep_Learning
## Overview
- This project focuses on Facial Emotion Recognition (FER) using deep learning models, with a particular emphasis on the FER 2013 dataset. 
- The objective is to build an emotion recognition system that can accurately classify facial expressions into seven distinct emotions: anger, disgust, fear, happiness, sadness, surprise, and neutral. 
- The project utilizes advanced deep learning techniques to preprocess data, design an effective neural network architecture, and optimize the model’s performance.

## Abstract
Facial Emotion Recognition (FER) is a rapidly growing field with applications in mental health diagnostics, human-computer interaction, and entertainment. This project explores the use of deep learning models to recognize emotions from facial expressions using the FER 2013 dataset. By leveraging convolutional neural networks (CNNs) and advanced training techniques, the project aims to improve model performance and robustness. 

## Key Features
- Dataset: The project uses the FER 2013 dataset from Kaggle, containing images of facial expressions labeled with seven emotions.
- Model Architecture: A five-layer deep neural network combining convolutional layers for feature extraction and fully connected layers for classification.
- Data Preprocessing: Includes resizing images to 48x48 pixels, normalizing pixel values, and encoding emotion labels into categorical values.
- Loss Function & Metrics: Categorical cross-entropy loss is used, with model performance evaluated based on accuracy.
- Handling Class Imbalance: Techniques like weighted loss and data augmentation (rotation, flipping) are applied to manage class imbalance and improve generalization.

## How This Helps?
This FER system has several potential applications:
- Mental Health: It can be used to monitor and assess emotional states, aiding in the diagnosis and tracking of mental health conditions.
- Human-Computer Interaction (HCI): Emotion-aware systems can improve interactions by understanding users' emotions and responding appropriately.
- Entertainment: Emotion recognition can be used to tailor content to users' emotional states, enhancing engagement and experience.

## Resources
- Packages: pandas, numpy, sklearn, matplotlib, seaborn tensorflow
- Dataset from Kaggle: https://www.kaggle.com/datasets/msambare/fer2013
- 
## Model Evaluation
- Accuracy: The model achieves an accuracy of 31.3% on the test set, reflecting its ability to classify facial expressions with reasonable success.
- Challenges: Despite using data augmentation and handling class imbalance, there is potential for further improvement in model performance.
- ![Confusion matrix](Confusion_Matrix.png)
## Future Directions: 
The model could be enhanced by exploring deeper architectures, using transfer learning from pretrained models, further tuning hyperparameters, and expanding the dataset.
