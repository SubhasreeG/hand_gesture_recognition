# Hand gesture recognition

## About the Project-
    This project implements basic hand gesture recognisation for numbers and alphabets using CNN. 
    The detected numbers are also generated as speech. The alphabets are spelled and word is generated as speech by demarkating them with a gesture for space.
    The system volume for the speech output can be increased or decreased with hand gestures.

## Requirements

opencv
tensorflow
keras
mediapipe
playsound == 1.2.2
gTTS
comtypes
pycaw

## Instructions for execution:
### To give custom labels:
	Change the lists: alphabet_label and number_label in utils/predict_result.py
### To train the model:
    Run training.ipynb
### To recognise numbers:
	python3 number_detection.py
### To recognise alphabet:
	python3 alphabet_detection.py
