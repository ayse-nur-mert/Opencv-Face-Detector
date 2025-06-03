import cv2
from deepface import DeepFace

# Start capturing video
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        continue

    # Ayna modu için görüntüyü yatay çevir
    frame = cv2.flip(frame, 1)

    try:
        # Perform emotion analysis
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        if result:
            # Get face coordinates and emotion
            face_coords = result[0]['region']
            emotion = result[0]['dominant_emotion']
            
            # Draw rectangle and emotion text
            x, y, w, h = face_coords['x'], face_coords['y'], face_coords['w'], face_coords['h']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    except:
        pass

    # Display the resulting frame
    cv2.imshow('Duygu Tanıma (Q: Çıkış)', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()

