import cv2
from fer import FER


detector = FER()


cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    if not ret:
        break

    
    emotions = detector.detect_emotions(frame)

    
    for emotion in emotions:
        (x, y, w, h) = emotion["box"]
        emotion_label = emotion["emotions"]
        dominant_emotion = max(emotion_label, key=emotion_label.get)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Emotion Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
