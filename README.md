# AgeGenderRace_CNN

This is a multi-classification project that aims to classify a person's age, gender and race based on their photo. The base model used was CNN.

Files included:
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

The dataset is the UTKFace taken from Kaggle: https://www.kaggle.com/datasets/jangedoo/utkface-new
It contains 20,000 images of faces of all ethnicities, ages and genders.

Project steps:
1. I created a separate model for each variable (age, gender, race).
  Age was categorized into 1-2, 3-9, 10-20, 21-27, 38-45, 46-65, and 65+.
  Gender was categorized into male and female.
  Race was categorized into White, Black, Asian, Indian, and Others.
2. The models were saved into their respective h5 files.
3. The final model "Predict Model.py" combined all three models to predict the age, gender and race of any photo inside the "test_input" file.
4. I used the Cascade Classifier to quickly identify faces on the image and discard non-faces.


How to use the code:
1. Import all relevant libraries. 
2. Load the UTKFace dataset.
3. Upload any image of a face into the "test_input" folder.
4. Copy the image path into "Predict Model.py" and get the face's age/gender/race predicted!
