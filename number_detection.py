from utils.predict_result import *
import cv2
import warnings
warnings.filterwarnings('ignore')
from gtts import gTTS
import os
from playsound import playsound  

import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from utils.hand_detection import HandDetector


handDetector = HandDetector(min_detection_confidence=0.7)

#Volume related initializations
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

## Capturing the video sequence
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

model = load_model('saved_model.h5')
aweight = 0.5
num_frames = 0
bg = None


def run_avg(img,aweight):
    global bg
    if bg is None:
        bg = img.copy().astype('float')
        return
    cv2.accumulateWeighted(img,bg,aweight)

def segment(img,thres=25):
    global bg
    diff = cv2.absdiff(bg.astype('uint8'),img)
    _, thresholded = cv2.threshold(diff,thres,255,cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(thresholded.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return
    else:
        segmented = max(contours,key = cv2.contourArea)
    return (thresholded,segmented)

num = 0
first_number = ""
operator = ""
second_number = ""
while(cap.isOpened()):
    num = num + 1
    ret, frame = cap.read()

    if ret ==True:
        frame = cv2.flip(frame, 1)
        (height, width) = frame.shape[:2]
        roi = frame[25:300, 25:300]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

         # ------------Detection of hand using mediapipe for changing volume-----------------------------------
        volume_frame=frame[0:1000,300:1000]
        
        handLandmarks = handDetector.findHandLandMarks(image=volume_frame, draw=True)

        if(len(handLandmarks) != 0):
                x1, y1 = handLandmarks[4][1], handLandmarks[4][2]
                x2, y2 = handLandmarks[8][1], handLandmarks[8][2]
                length = math.hypot(x2-x1, y2-y1)
                print(length)

                volumeValue = np.interp(length, [50, 250], [-65.25, 0.0]) #coverting length to proportionate to volume range
                volume.SetMasterVolumeLevel(volumeValue, None)

                cv2.circle(volume_frame, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(volume_frame, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
                cv2.line(volume_frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

        # -----------------------------------------------------------------------------------------

        if num_frames < 30:
            run_avg(gray, aweight)
        else:
            hand = segment(gray)

            if hand is not None:
                (thresholded, segmented) = hand
                cv2.drawContours(frame, [segmented + (25, 25)], -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)
                print("Some part of hand is detecting")
                contours, _= cv2.findContours(thresholded,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

                for cnt in contours:
                    if cv2.contourArea(cnt) > 5000:
                        print("Hand detecting for prediction")
                        gesture, probability = get_prediction(thresholded)
                        text = "{}: {:.2f}%".format(gesture, probability * 100)

                        myobj = gTTS(text=str(gesture), lang='en', slow=True)
    
                        if num%5 == 0:
                            myobj.save("audio_output.mp3")
                            playsound("audio_output.mp3")
                            os.remove("audio_output.mp3")

                        cv2.putText(frame, text, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,	0.7, (0, 255, 0), 2)

                        #cv2.imshow('Press  ESC ',frame)

        cv2.rectangle(frame, (25, 25), (300, 300), (0, 255, 0), 2)
        cv2.putText(frame, "Hand gesture recognition", (200, 20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 30, 255), 3)

        num_frames += 1


        cv2.imshow('frame', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()

print(first_number, operator, second_number)
cv2.waitKey()
cv2.destroyAllWindows()