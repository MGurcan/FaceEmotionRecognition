# Summary
This repository is dedicated to the exploration of various computer vision techniques for Facial Emotion Recognition (FER). We have implemented and evaluated three distinct methods to determine their effectiveness in recognizing facial emotions. Detailed below are the instructions and resources to access the test and train notebooks for each method.

---

# Final Test Notebook
/FaceEmotionProject/Scripts/ModelEducation/Methods/Final_Test/TestModels.ipynb

#### For test images, a sample set of test images is provided below this folder.
/Users/gurcan/Desktop/School/Bil468/FaceEmotionProject/Scripts/ModelEducation/Methods/Final_Test/ImageSample

By making run all call all model's test outputs can be show. In additionally all three methods' test sections divided appropriately. If you want to test on a different image, it can be added to the Image Sample folder.

Below the README file drive link which includes all data will be provided.

---

# Final Train Notebooks
Method-3's train part processed via Google-Colab. If it needs to be tested on local just change file paths into your existing data files.
## Method-1
/FaceEmotionProject/Scripts/ModelEducation/Methods/Method1/Final_Train/method1_copy.ipynb

## Method-2
/FaceEmotionProject/Scripts/ModelEducation/Methods/Method2/Final_Train/method2_copy.ipynb

## Method-3
/FaceEmotionProject/Scripts/ModelEducation/Methods/Method3/Final_Train/BaseCNN_VGG19_ResNET_train.ipynb

# Drive Link (Public Access)
https://drive.google.com/drive/folders/1nCjf3n_J1JopZ6Ppa37441bIXXtLEwGx?usp=sharing

---

# Usage Instructions
### Clone the Repository: Start by cloning this repository to your local machine or open it directly in Google Colab.

### Navigate to Test/Train Notebooks: Use the paths provided above to access the training and testing notebooks for each method.

### Running the Notebooks: Execute the cells in the notebooks to train the models or predict using the pre-trained models. Make sure to adjust the paths if you are working locally.

### Adding Test Images: You can add more images to the ImageSample folder to test the models' performance on new data.

### Viewing Outputs: The outputs for each test can be viewed directly in the notebooks, allowing for easy comparison and analysis of the models' performance.

---

# Try Website Version of VGG19

- run npm install in Website / emotion-recognition folder
- then run npm start on Website
- for VGG service run python3 Vgg19.py on Website / emotion-recognition / server / Models folder
- After that you will be able to try emotion recognition by VGG19 with webcam images