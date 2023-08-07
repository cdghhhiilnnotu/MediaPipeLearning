import cv2
import mediapipe as mp
import time




class FaceDetector():
    def __init__(self, minDectectionCon=0.5):
        self.minDectectionCon = minDectectionCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDectectionCon)

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxes = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                ### 1
                # mpDraw.draw_detection(img, detection)
                ### 2
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bboxes.append([id, bbox, detection.score])
                if draw:
                    img = self.fancyDraw(img, bbox) 
                    cv2.putText(img, str(int(detection.score[0]*100)), (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)
        
        return img, bboxes
    
    def fancyDraw(self, img, bbox, l=30, t=10, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, (255, 0, 0), rt)
        cv2.line(img, (x,y), (x+l, y), (255,0,255), t)
        cv2.line(img, (x,y), (x, y+l), (255,0,255), t)

        return img


def main():
    cap = cv2.VideoCapture('Video1.mp4')
    pTime = 0

    detector = FaceDetector()

    while True:
        success, img = cap.read()
        img, bboxes = detector.findFaces(img)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)

        cv2.imshow('Image', img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()

















