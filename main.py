import cv2
import os
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.known_face_names = []
        self.trained = False

    def load_encoding_images(self, images_path):
        face_encodings = []
        
        # Her bir kişi klasörünü işle
        for person_folder in os.listdir(images_path):
            person_path = os.path.join(images_path, person_folder)
            if not os.path.isdir(person_path):
                continue
                
            print(f"Kişi klasörü işleniyor: {person_folder}")
            
            # Kişinin tüm fotoğraflarını işle
            for img_name in os.listdir(person_path):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                full_path = os.path.join(person_path, img_name)
                img = cv2.imread(full_path)
                if img is None:
                    print(f"Hata: {full_path} okunamadı!")
                    continue
                    
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    face = gray[y:y+h, x:x+w]
                    face_encodings.append(face)
                    self.known_face_names.append(person_folder)
                    print(f"{person_folder} için yüz bulundu ve eklendi ({img_name})")
                else:
                    print(f"Uyarı: {full_path} içinde yüz bulunamadı!")
        
        if face_encodings:
            faces_array = []
            labels = []
            for i, face in enumerate(face_encodings):
                face_resized = cv2.resize(face, (100, 100))
                faces_array.append(face_resized)
                labels.append(i)
            
            self.face_recognizer.train(faces_array, np.array(labels))
            self.trained = True
            print(f"\nToplam {len(self.known_face_names)} yüz tanındı.")
            print("Tanınan kişiler:", list(set(self.known_face_names)))
        else:
            print("Hiç yüz bulunamadı!")

    def detect_known_faces(self, frame):
        if not self.trained:
            return [], []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_locations = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        face_names = []
        
        for (x, y, w, h) in face_locations:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (100, 100))
            
            try:
                label, confidence = self.face_recognizer.predict(face_resized)
                if confidence < 100:  # Güven eşiği
                    name = self.known_face_names[label]
                    confidence_text = name  # Sadece ismi göster
                else:
                    confidence_text = "Bilinmiyor"
                face_names.append(confidence_text)
            except Exception as e:
                print(f"Hata: {e}")
                face_names.append("Bilinmiyor")
        
        return face_locations, face_names

# Ana program
def main():
    # Yüz tanıma sistemini başlat
    sfr = SimpleFacerec()
    
    # Yüzleri yükle ve modeli eğit
    print("Eğitim verileri yükleniyor...")
    sfr.load_encoding_images("known_faces")
    
    if not sfr.trained:
        print("Hata: Model eğitilemedi!")
        return
        
    # Kamerayı başlat
    print("\nKamera başlatılıyor...")
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Yüzleri tespit et
        face_locations, face_names = sfr.detect_known_faces(frame)
        
        # Sonuçları göster
        for (x, y, w, h), name in zip(face_locations, face_names):
            if "Bilinmiyor" in name:
                color = (0, 0, 255)  # Kırmızı
            else:
                color = (0, 255, 0)  # Yeşil
                
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x, y-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        
        cv2.imshow("Yuz Tanima", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
