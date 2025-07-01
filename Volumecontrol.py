import cv2
import numpy as np
import time
import handtrackingmodule as htm
import os
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
volume.SetMasterVolumeLevel(-20.0, None)
minvol=volume.GetVolumeRange()[0]
maxvol=volume.GetVolumeRange()[1]
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs
wcam=640
hcam=480
video = cv2.VideoCapture(1)
video.set(3,wcam)
video.set(4,hcam)
ptime=0
volbar=150
detector=htm.HandDetector()
while True:
    success, img = video.read()
    ctime= time.time()
    img=detector.find_hands(img)
    lmlist=detector.find_position(img)
    if len(lmlist) > 8:  # Check if landmarks exist
        x1, y1 = lmlist[4][1], lmlist[4][2]  # Thumb tip
        x2, y2 = lmlist[8][1], lmlist[8][2]  # Index finger tip
        cx,cy=(x1+x2)//2,(y1+y2)//2
        length= np.hypot(x2-x1,y2-y1)
        cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
        # print(f"Thumb Tip: ({x1}, {y1}), Index Finger Tip: ({x2}, {y2})")
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)
        #hand range 50-300
        #volume range -65 to 0'
        vol= np.interp(length,[50,300],[minvol,maxvol])
        volbar= np.interp(length,[50,300],[400,150])
        volper= np.interp(length,[50,300],[0,100])
        volume.SetMasterVolumeLevel(vol, None)
        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (50, int(volbar)), (85, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'Volume: {int(volper)}%', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)


    fps= 1/(ctime-ptime)
    ptime=ctime
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 3)
    cv2.imshow("Image", img)
    q=cv2.waitKey(1)
    if q==ord('q'):
        break