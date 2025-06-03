import cv2
import os
import numpy as np

# Ensure cv2.face is available
if not hasattr(cv2, 'face'):
    raise ImportError("cv2.face modülü bulunamadı. Lütfen 'opencv-contrib-python' paketini yükleyin.")

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def prepare_training_data(data_folder_path):
    if not os.path.exists(data_folder_path):
        print(f"'{data_folder_path}' klasörü bulunamadı. Lütfen önce yüz kaydedin.")
        return None, None, None

    faces, labels = [], []
    label_map = {}
    
    for image_name in os.listdir(data_folder_path):
        if not image_name.endswith(('.jpg', '.jpeg', '.png')):
            continue

        # Kişi adını dosya adından al (timestamp'i kaldır)
        person_name = image_name.split('_')[0]
        
        # Kişiye numara ata
        if person_name not in label_map:
            label_map[len(label_map)] = person_name
            
        label = [k for k, v in label_map.items() if v == person_name][0]
        
        # Resmi yükle ve yüzü tespit et
        image_path = os.path.join(data_folder_path, image_name)
        img = cv2.imread(image_path)
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        if len(face_rect) == 1:
            x, y, w, h = face_rect[0]
            face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            faces.append(face)
            labels.append(label)

    if not faces:
        return None, None, None
        
    return faces, labels, label_map

def main():
    # Load training faces
    data_path = 'faces'
    result = prepare_training_data(data_path)
    if result[0] is None:
        return

    faces, labels, label_map = result
    print(f"Eğitim için {len(faces)} yüz yüklendi.")

    # Create and train LBPH recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    # Sabit eşik değeri kullan
    threshold = 80  # Daha düşük değer = daha kesin eşleşme

    # Start video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılamadı.")
        return
    
    print("Yüz tanıma başlatıldı. Çıkmak için 'q' tuşuna basın.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Ayna modu için görüntüyü yatay çevir
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces_rects:
            face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            label, confidence = recognizer.predict(face)

            if confidence < threshold and label in label_map:
                name = label_map[label]
                color = (0, 255, 0)  # Yeşil
            else:
                name = 'Bilinmiyor'
                color = (0, 0, 255)  # Kırmızı

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow('Yuz Tanima (Q: Cikis)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
