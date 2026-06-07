from flask import Flask, render_template, Response
from keras.models import load_model
import numpy as np
import cv2
from threading import Thread
import signal
import sys
from pygame import mixer
import os

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
path = os.getcwd()

mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

lbl = ['Close', 'Open']

model = load_model('models/cnnCat2.keras')
path = os.getcwd()

app = Flask(__name__)

cap = None
is_camera_running = False

def start_camera():
    global cap, thicc
    cap = cv2.VideoCapture(0)
    count = 0
    score = 0
    thicc = 2  
    lbl = ['Close', 'Open']
    while is_camera_running:
        ret, frame = cap.read()
        if not ret:
            break
        height, width = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
        left_eye = leye.detectMultiScale(gray)
        right_eye = reye.detectMultiScale(gray)

        cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

        for (x, y, w, h) in right_eye:
            r_eye = frame[y:y + h, x:x + w]
            count += 1
            r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye, (24, 24))
            r_eye = r_eye / 255
            r_eye = r_eye.reshape(24, 24, -1)
            r_eye = np.expand_dims(r_eye, axis=0)
            rpred = np.argmax(model.predict(r_eye), axis=1)

            if rpred[0] == 0:  
                lbl = 'Closed'
            else:
                lbl = 'Open'

        for (x, y, w, h) in left_eye:
            l_eye = frame[y:y + h, x:x + w]
            count += 1
            l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
            l_eye = cv2.resize(l_eye, (24, 24))
            l_eye = l_eye / 255
            l_eye = l_eye.reshape(24, 24, -1)
            l_eye = np.expand_dims(l_eye, axis=0)
            lpred = np.argmax(model.predict(l_eye), axis=1)

            if lpred[0] == 0:  
                lbl = 'Closed'
            else:
                lbl = 'Open'

        if rpred[0] == 0 and lpred[0] == 0:
            score = score + 1
            cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            score = score - 1
            cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        if score < 0:
            score = 0
        cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        if score >= 15:
           
            cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
            try:
                sound.play()
            except:
                pass

            if thicc < 16:
                thicc = thicc + 2
            else:
                thicc = thicc - 2
                if thicc < 2:
                    thicc = 2

            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

        
        cv2.imshow("Drowsiness Detection", frame)
        cv2.waitKey(1)

    if cap is not None:
        cap.release()

@app.route('/')
def index():
    return render_template('index.html')

def generate():
    global cap
    while is_camera_running:
        ret, frame = cap.read()
        if not ret:
            break

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera_route():
    global is_camera_running
    is_camera_running = True
    Thread(target=start_camera).start()
    return "Camera Started"

def signal_handler(sig, frame):
    global is_camera_running
    is_camera_running = False
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    app.run(debug=True)