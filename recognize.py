import cv2
import os
from datetime import datetime

# Yüz tespiti için cascade sınıflandırıcı
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# faces klasörünü kontrol et ve oluştur
os.makedirs("faces", exist_ok=True)

# Kamera başlat
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kamera açılamadı!")
    exit(1)

# Kullanıcıdan isim al
name = input("Lütfen adınızı girin: ").strip()
if not name:
    print("İsim boş olamaz!")
    exit(1)

print("Yüzünüzü kaydetmek için SPACE tuşuna, çıkmak için Q tuşuna basın.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
        
    # Ayna modu için görüntüyü yatay çevir
    frame = cv2.flip(frame, 1)
        
    # Gri tonlamaya çevir
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Yüzleri tespit et
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Yüz bölgesini dikdörtgen içine al
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Görüntüyü göster
    cv2.imshow('Yuz Kayit - SPACE: Kaydet, Q: Cikis', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("İşlem iptal edildi.")
        break
    elif key == ord(' '):  # Space tuşuna basıldığında
        if len(faces) == 1:  # Sadece bir yüz varsa
            face_img = frame[y:y+h, x:x+w]  # Yüz bölgesini kes
            
            # Benzersiz dosya adı oluştur
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join("faces", f"{name}_{timestamp}.jpg")
            
            # Yüzü kaydet
            cv2.imwrite(file_path, face_img)
            print(f"Yüz başarıyla kaydedildi: {file_path}")
            break
        else:
            print("Lütfen kameraya tek bir yüz gösterin!")

# Temizlik
cap.release()
cv2.destroyAllWindows()