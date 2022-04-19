import cv2

# download the cascades
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")
# for getting an error of cv2 has no attribute casacadeclassifier use pip install opencv-python


# load the cascades
# define a function to detect

def detect_face_eyes(gray_image, org_image):
    faces = face_cascade.detectMultiScale(gray_image, 1.2, 5)
    # gray_image, scale, neighbors
    for (x, y, w, h) in faces:
        #   draw rectangle
        cv2.rectangle(org_image, (x, y), (x + w, y + h), (255, 0, 0), 3)
        # org_image, TL corner, BR corner, scale
        # create regions of interest - org image, gray image as eyes are to be found in reference to
        # face in our case
        # face = [y:y+h, x:x+w]
        roi_gray = gray_image[y:y + h, x:x + w]
        roi_org = org_image[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(gray_image, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(org_image, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)

    return org_image


# get the webcam to get the images

video_loop = cv2.VideoCapture(0)

while True:
    _, frame = video_loop.read()

    # convert the image to gray
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    anno_image = detect_face_eyes(image, frame)
    cv2.imshow("Annotated Video Loop", anno_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_loop.release()
cv2.destroyAllWindows()

# release the video device
# destroy all windows
