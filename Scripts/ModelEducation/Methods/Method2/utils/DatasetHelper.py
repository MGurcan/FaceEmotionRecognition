import numpy as np
import os
import cv2
import shutil
from sklearn.model_selection import train_test_split

def prepare_small_test_set(X_test, y_test):
    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    X_test_small, y_test_small = [], []
    for classname in classes:
        X_test_small.append(X_test[y_test == classname][:200])
        y_test_small.append(y_test[y_test == classname][:200])

    X_test_small = np.concatenate(X_test_small)
    y_test_small = np.concatenate(y_test_small)
    return X_test_small, y_test_small

def load_images_from_folder_fer_2013(folder, max_images_per_class=None, flatten=True):
    images = []
    labels = []
    for label_folder in os.listdir(folder):
        label_folder_path = os.path.join(folder, label_folder)
        if os.path.isdir(label_folder_path):
            count_images_for_class = 0
            print("label_folder_path: ", label_folder_path)
            for image_file in os.listdir(label_folder_path):
                if max_images_per_class is not None and count_images_for_class >= max_images_per_class:
                    print(f"Reached maximum number of images for class {label_folder}. Skipping the rest.")
                    break
                count_images_for_class += 1
                image_path = os.path.join(label_folder_path, image_file)
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (48, 48))
                if flatten:
                    images.append(img.flatten())
                else:
                    images.append(img)
                labels.append(label_folder)
    return np.array(images), np.array(labels)

def load_images_from_folder_expW(folder, max_images_per_class=None):
    images = []
    labels = []
    for label_folder in os.listdir(folder):
        label_folder_path = os.path.join(folder, label_folder)
        if os.path.isdir(label_folder_path):
            count_images_for_class = 0
            print("label_folder_path: ", label_folder_path)
            for image_file in os.listdir(label_folder_path):
                if max_images_per_class is not None and count_images_for_class >= max_images_per_class:
                    print(f"Reached maximum number of images for class {label_folder}. Skipping the rest.")
                    break
                count_images_for_class += 1
                image_path = os.path.join(label_folder_path, image_file)
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (100, 100))
                images.append(img.flatten())
                labels.append(label_folder)
    return np.array(images), np.array(labels)


def load_images_from_folder_fer_2013_blurred(folder):
    images = []
    labels = []
    for label_folder in os.listdir(folder):
        label_folder_path = os.path.join(folder, label_folder)
        if os.path.isdir(label_folder_path):
            for image_file in os.listdir(label_folder_path):
                image_path = os.path.join(label_folder_path, image_file)
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (48, 48))
                img = cv2.GaussianBlur(img, (5, 5), 0)
                images.append(img.flatten())
                labels.append(label_folder)
    return np.array(images), np.array(labels)


def histogram_equalization(image):
    # plot the image
    equalized_image = cv2.equalizeHist(image)
    return equalized_image

def load_images_from_folder_fer_2013_histogram_equilazed(folder):
    images = []
    labels = []
    for label_folder in os.listdir(folder):
        label_folder_path = os.path.join(folder, label_folder)
        if os.path.isdir(label_folder_path):
            for image_file in os.listdir(label_folder_path):
                image_path = os.path.join(label_folder_path, image_file)
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = histogram_equalization(img)
                img = cv2.resize(img, (48, 48))
                images.append(img.flatten())
                labels.append(label_folder)
    return np.array(images), np.array(labels)

def load_images_from_folder_fer_2013_edge_boosted(folder):
    images = []
    labels = []
    for label_folder in os.listdir(folder):
        label_folder_path = os.path.join(folder, label_folder)
        if os.path.isdir(label_folder_path):
            for image_file in os.listdir(label_folder_path):
                image_path = os.path.join(label_folder_path, image_file)
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
  
                # Sharpen the image 
                sharpened_image = cv2.filter2D(img, -1, kernel)

                img = cv2.resize(sharpened_image, (48, 48))
                images.append(img.flatten())
                labels.append(label_folder)
    return np.array(images), np.array(labels)

def load_images_from_folder_fer_2013_edge_boosted_histogram_equilazed(folder):
    images = []
    labels = []
    for label_folder in os.listdir(folder):
        label_folder_path = os.path.join(folder, label_folder)
        if os.path.isdir(label_folder_path):
            for image_file in os.listdir(label_folder_path):
                image_path = os.path.join(label_folder_path, image_file)
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = histogram_equalization(img)
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
  
                # Sharpen the image 
                sharpened_image = cv2.filter2D(img, -1, kernel)
                img = cv2.resize(sharpened_image, (48, 48))

                img = cv2.resize(img, (48, 48))
                images.append(img.flatten())
                labels.append(label_folder)
    return np.array(images), np.array(labels)


def preprocess_expw_images(folder, target_folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                # Dosya yolu oluşturma
                file_path = os.path.join(root, file)

                # Görüntüyü okuma ve griye çevirme
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

                # Görüntüyü yeniden boyutlandırma
                img_resized = cv2.resize(img, (100, 100))

                # Hedef klasör yolu oluşturma
                relative_path = os.path.relpath(root, folder)
                target_path = os.path.join(target_folder, relative_path)

                # Hedef klasörde aynı hiyerarşi oluşturma
                if not os.path.exists(target_path):
                    os.makedirs(target_path)

                # Görüntüyü hedef klasöre kaydetme
                cv2.imwrite(os.path.join(target_path, file), img_resized)


def split_data(source_folder, train_folder, validation_folder, test_size=0.2):
    # Duygu kategorilerini listele
    categories = os.listdir(source_folder)

    for category in categories:
        category_path = os.path.join(source_folder, category)

        # Her bir kategori için dosya isimlerini listele
        files = [file for file in os.listdir(
            category_path) if file.lower().endswith(('png', 'jpg', 'jpeg'))]

        # Dosyaları eğitim ve doğrulama setlerine ayır
        train_files, validation_files = train_test_split(
            files, test_size=test_size, random_state=42)

        # Eğitim dosyalarını kopyala
        train_category_path = os.path.join(train_folder, category)
        if not os.path.exists(train_category_path):
            os.makedirs(train_category_path)
        for file in train_files:
            shutil.copy(os.path.join(category_path, file),
                        os.path.join(train_category_path, file))

        # Doğrulama dosyalarını kopyala
        validation_category_path = os.path.join(validation_folder, category)
        if not os.path.exists(validation_category_path):
            os.makedirs(validation_category_path)
        for file in validation_files:
            shutil.copy(os.path.join(category_path, file),
                        os.path.join(validation_category_path, file))


def merge_datasets(dataset1_path, dataset2_path, merged_dataset_path):
    for dataset_path in [dataset1_path, dataset2_path]:
        for split in ['train', 'validation']:
            current_split_path = os.path.join(dataset_path, split)
            
            categories = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            # Duygu kategorilerini listele
            # categories = os.listdir(current_split_path)
            
            for category in categories:
                category_path = os.path.join(current_split_path, category)
                files = [file for file in os.listdir(category_path) if file.lower().endswith(('png', 'jpg', 'jpeg'))]
                
                # Hedef klasör yolu
                target_category_path = os.path.join(merged_dataset_path, split, category)
                
                # Hedef klasörde aynı hiyerarşi oluşturma
                if not os.path.exists(target_category_path):
                    os.makedirs(target_category_path)
                
                # Dosyaları hedef klasöre kopyalama
                for file in files:
                    source_file_path = os.path.join(category_path, file)
                    target_file_path = os.path.join(target_category_path, file)
                    
                    # Dosyanın zaten var olup olmadığını kontrol et
                    if not os.path.exists(target_file_path):
                        shutil.copy(source_file_path, target_file_path)
                    else:
                        print(f"File {file} already exists in {target_category_path}. Skipping.")
