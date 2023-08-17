import cv2
import numpy as np
import PoseEstimationModule as pm
import time

cap = cv2.VideoCapture("D:\\GitWorking\\OpenCV-Learning\\Assets\\video1.mp4")

detector = pm.PoseDetector()
count = 0
dir = 0
pTime = 0

while True:
    success, img = cap.read()
    img = detector.findPose(img, False)
    lmList = detector.getPosition(img, False)

    if len(lmList) != 0:
        # detector.findAngle(img, 12, 14, 16)
        angle = detector.findAngle(img, 11, 13, 15)
        per = np.interp(angle, (210, 310), (0, 100))
        bar = np.interp(angle, (220, 310), (650, 100))
        # print(per, angle)

        if per == 100:
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            if dir == 1:
                count += 0.5
                dir = 0

        cv2.rectangle(img, (400, 100), (475,  650), (0, 255, 0), 3)
        cv2.rectangle(img, (400, int(bar)), (475, 650), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(per)}', (400, 75), cv2.FONT_HERSHEY_PLAIN, 4,
                    (255,0,0), 4)

        cv2.rectangle(img, (0,450), (250, 720), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(count)}', (45, 670), cv2.FONT_HERSHEY_PLAIN, 15,
                    (255,0,0), 25)
        
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'{int(fps)}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 5,
                    (255,0,0), 5)

    cv2.imshow('Image', img)
    cv2.waitKey(1)












