import cv2
import numpy as np
from keras.models import model_from_json
import math

def load_rgba_image(path):
    # Görüntüyü RGBA olarak yükle
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image.shape[2] != 4:  # Eğer görüntü 4 kanallı değilse
        # Görüntüyü RGBA'ya dönüştür
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    return image

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
emotion_emojis = {
    "Angry": ("./emojis/angry.jpg", (255, 0, 0)),  # Kırmızı
    "Disgusted": ("./emojis/disgusted.jpg", (0, 255, 0)),  # Yeşil
    "Fearful": ("./emojis/angry.jpg", (128, 0, 128)),  # Mor
    "Happy": ("./emojis/angry.jpg", (255, 255, 0)),  # Sarı
    "Neutral": ("./emojis/angry.jpg", (128, 128, 128)),  # Gri
    "Sad": ("./emojis/angry.jpg", (0, 0, 255)),  # Mavi
    "Surprised": ("./emojis/surprised.png", (0, 255, 255))  # Turuncu
}
# load json and create model
json_file = open('./emotion_model_30epoch.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("./emotion_model_30epoch.h5")
print("Loaded model from disk")

# start the webcam feed
cap = cv2.VideoCapture(0)

# pass here your video path
# you may download one from here : https://www.pexels.com/video/three-girls-laughing-5273028/
#cap = cv2.VideoCapture("C:\\JustDoIt\\ML\\Sample_videos\\emotion_sample6.mp4")

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        maxindex = int(np.argmax(emotion_prediction))
        emotion = emotion_dict[maxindex]
        emoji_path, color = emotion_emojis[emotion]

        # Arka plan rengini değiştir
        overlay = frame.copy()
        overlay[:,:] = color
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)

        # Emojiyi yüzün yanına yerleştirin
        # emoji = cv2.imread(emoji_path, -1)

        small_h = math.ceil(h/5)
        small_w = math.ceil(w/5)
        emoji = load_rgba_image(emoji_path)
        emoji = cv2.resize(emoji, (small_h, small_w))
        for i in range(0, small_h):
            for j in range(0, small_w):
                if emoji[i, j][3] != 0:  # Alpha değeri 0 olmayan pikseller
                    frame[y + i, x + j] = emoji[i, j][:3]
        # Emojiyi ekranda göster
        #emoji_img = cv2.imread(emoji_path)
        #emoji_img = cv2.resize(emoji_img, (w, h))
        #frame[y:y+h, x:x+w] = emoji_img

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
