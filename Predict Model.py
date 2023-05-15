#!/usr/bin/env python
# coding: utf-8

# In[14]:


from keras.models import load_model
from PIL import Image
import numpy as np
import cv2

import tensorflow as tf

#the following are to do with this interactive notebook code

get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt # this lets you draw inline pictures in the notebooks
import pylab # this allows you to control figure size 
pylab.rcParams['figure.figsize'] = (10.0, 8.0) # this controls figure size in the notebook


# # Loading Age/Gender/Race Models

# In[6]:


age_model = load_model(r"age_output/Output/cnn_logs/age_model_final_save.h5")

# summarize model.
age_model.summary()


# In[7]:


gender_model = load_model(r"gender_output/cnn_logs/gender_model.h5")

# summarize model.
gender_model.summary()


# In[165]:


race_model = load_model(r"race_output/cnn_logs/race_model_final_saveA.h5")
# summarize model.
race_model.summary()


# In[166]:


# Labels on Age, Gender and Race to be predicted

age_ranges = ['1-2', '3-9', '10-20', '21-27', '28-45', '46-65', '66-116']
gender_ranges = ['male', 'female']
race_ranges= ['white','black','asian', 'indian', 'others']


# # Loading Test Images

# In[178]:


# Song Hye Kyo - Female - Asian - 41 years old
#img_path = (r"test_input/images/Screenshot 2023-05-13 164811.jpg")

# Daniel Kaluuya - Male - Black - 34 years old
#img_path = (r"test_input/images/Screenshot 2023-05-13 171159.jpg")

# Jenna Ortega - Female - White - 20 years old
#img_path = (r"test_input/images/Screenshot 2023-05-13 165620.jpg")

# Antonia Gentry - Female - Biracial (White/Black) - 25 years old
#img_path = (r"test_input/images/Screenshot 2023-05-13 165731.jpg")

# Priyanka Chopra - Female - Indian - 40 years old
#img_path = (r"test_input/images/priyanka_chopra.jpg")

# Pair photo
img_path = (r"test_input/images/Screenshot 2023-05-13 170357.jpg")

# Group photo
#img_path = (r"test_input/images/Screenshot 2023-05-13 170049.jpg")


# In[179]:


from IPython.display import Image 
pil_img = Image(filename=img_path)
display(pil_img)


# # Results on Age/Gender/Race

# In[180]:


tf.debugging.disable_traceback_filtering


# In[181]:


test_image = cv2.imread(img_path)
gray = cv2.cvtColor(test_image,cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier(r"test_input/cv2_cascade_classifier/haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

i = 0

for (x,y,w,h) in faces:
    i = i+1
    cv2.rectangle(test_image,(x,y),(x+w,y+h),(203,12,255),2)

    img_gray=gray[y:y+h,x:x+w]
  
    gender_img = cv2.resize(img_gray, (100, 100), interpolation = cv2.INTER_AREA)
    gender_image_array = np.array(gender_img)
    gender_input = np.expand_dims(gender_image_array, axis=0)
    output_gender=gender_ranges[np.argmax(gender_model.predict(gender_input))]

    age_image=cv2.resize(img_gray, (200, 200), interpolation = cv2.INTER_AREA)
    age_input = age_image.reshape(-1, 200, 200, 1)
    output_age = age_ranges[np.argmax(age_model.predict(age_input))]
    
    race_img = cv2.resize(img_gray, (100, 100), interpolation = cv2.INTER_AREA)
    race_image_array = np.array(race_img)
    race_input = np.expand_dims(race_image_array, axis=0)
    output_race= race_ranges[np.argmax(race_model.predict(race_input))]
    
    output_str = str(i) + ": "+  output_gender + ', '+ output_age + ', '+ output_race 
    print(output_str)
  
    col = (0,255,0)

    cv2.putText(test_image, str(i),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,col,2)

plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))


# In[ ]:




