import cv2

cascade_path = "haarcascades/haarcascade_smile.xml"
smile_cascade = cv2.CascadeClassifier(cascade_path)
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")

def detect_smile(gray_image, org_image):
    # smile = smile_cascade.detectMultiScale(gray_image, 1.2, 5)
    # for (x, y, w, h) in smile:
    #     cv2.rectangle(org_image, (x, y), (x + w, y + h), (255, 0, 255), 2)
    #
    # return org_image
    faces = face_cascade.detectMultiScale(gray_image, 1.2, 5)
    # gray_image, scale, neighbors
    for (x, y, w, h) in faces:
        #   draw rectangle
        cv2.rectangle(org_image, (x, y), (x + w, y + h), (255, 0, 0), 3)
        roi_gray = gray_image[y:y + h, x:x + w]
        roi_org = org_image[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_org, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)

        smile = smile_cascade.detectMultiScale(roi_gray, 1.1, 30)
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_org, (sx, sy), (sx + sw, sy + sh), (255, 0, 255), 2)
    return org_image



video = cv2.VideoCapture(0)
while True:
    _, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    anno_image = detect_smile(gray, frame)

    cv2.imshow("Smile face", anno_image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
