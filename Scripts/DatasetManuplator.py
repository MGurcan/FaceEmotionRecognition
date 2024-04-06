import os
import shutil

source_dir = '../Data/Expw-F'
destination_dir = '../Data/Expw-F_sample500'

if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

emotions = os.listdir(source_dir)

for emotion in emotions:
    emotion_source_dir = os.path.join(source_dir, emotion)
    emotion_destination_dir = os.path.join(destination_dir, emotion)
    
    if not os.path.exists(emotion_destination_dir):
        os.makedirs(emotion_destination_dir)
    
    images = os.listdir(emotion_source_dir)
    
    for image in images[:500]:
        source_image_path = os.path.join(emotion_source_dir, image)
        destination_image_path = os.path.join(emotion_destination_dir, image)
        shutil.copy(source_image_path, destination_image_path)

print("İşlem tamamlandı!")
