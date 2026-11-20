# Deeplearning--Emotion-Detection
A model trained with self-made dataset, for detecting emotions.
The aim of this project is to create a real time facial emotion recognition system by implementing two PyTorch models: ResNet18 for emotion classification and YOLOv8 with OpenCV for face detection via a webcam.

The model was trained on a small, custom-collected data set of images of faces of friends expressing three emotions: happy, sad, and angry.

The training data were specifically composed with females. Other webcam smoke tests were performed with both male and female subjects to check for generalization of the model. The model was able to predict in both groups, but there were some differences in the confidence level and performance of the predictions.

The goal of this project is to investigate transfer learning, facial emotion classification, real time inference and the problem of generalisation from a small, demographically restricted data set.
