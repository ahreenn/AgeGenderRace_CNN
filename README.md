# AgeGenderRace_CNN

This is a multi-classification project that aims to classify a person's age, gender and race based on their photo using CNN and Tensorflow. Some difficulties were encountered when creating the final "Predict Model.py" to merge all three models together in the coding aspect. After injecting external images into the model, the final results looked more promising on age and gender - however, the model seems to predict race quite inaccurately - perhaps due to the photos being transformed into greyscale.


# Files included:
- age_output
- gender_output
- race_output
- test_input
- Age Model Final.py
- Gender Model Final.py
- Predict Model.py
- README.md
- Race Model Final.py
- requirements.txt


# Download the dataset
The dataset is the UTKFace taken from Kaggle: https://www.kaggle.com/datasets/jangedoo/utkface-new
It contains 20,000 images of faces of all ethnicities, ages and genders.
*Please save it under the AgeGenderRace_CNN folder.


# Project steps:
4. A separate CNN model for each variable (age, gender, race) into different .py files.
  Age was categorized into 1-2, 3-9, 10-20, 21-27, 38-45, 46-65, and 65+.
  Gender was categorized into male and female.
  Race was categorized into White, Black, Asian, Indian, and Others.
2. The models were saved into their respective h5 files.
3. The final model "Predict Model.py" combined all three models to predict the age, gender and race of any photo inside the "test_input" file.
4. I used the Cascade Classifier to quickly identify faces on the image and discard non-faces.


# How to use model:
1. Open "Predict Model.py"
2. Import all relevant libraries. 
3. Upload any image of a face into the "test_input" folder.
4. Copy the image path into "Predict Model.py" and get the face's age/gender/race predicted!
