import cv2

# pre_trained Data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_data = cv2.CascadeClassifier('haarcascade_smile.xml')

# to capture webcam.

webcam = cv2.VideoCapture(0)
# 0 FOR video capture can use video instead of 0 ('video.mp4')

while True:
    successful_frame_read, frame = webcam.read()

    if not successful_frame_read:
        break

    # convert video/ webcam to grayscale
    grayscale_vid = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cord = trained_face_data.detectMultiScale(grayscale_vid)

    # draw Rectangles around the faces
    for (x, y, w, h) in face_cord:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # (img, (TopLeft cord), (cord+width), (BGR col), thickness)

        the_face = frame[y:y + h, x:x + w]
        grayscale_face = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        smile_cord = smile_data.detectMultiScale(grayscale_face, scaleFactor=1.7, minNeighbors=20)
        if len(smile_cord) > 0:
            cv2.putText(frame, 'Smiling', (x, y + h + 40), fontScale=2, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        color=(255, 255, 255))

    cv2.imshow('webcam smile detector', frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break
