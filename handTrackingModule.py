import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self,mode=False,maxHands=2,detectioncon=0.5,trackingcon=0.5):
        self.mode = mode
        self.maxhands = maxHands
        self.detectioncon = detectioncon
        self.trackingcon = trackingcon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxhands,self.detectioncon,self.trackingcon)
        self.mpdraw = mp.solutions.drawing_utils

    def findhands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)
        # this is for check multi hands
        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:
                # showing hand landmarks AND ITS connections on original image
                if draw:
                    self.mpdraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img


    def findPosition(self,img, handNo=0, draw=False):
        lmList = []
        if self.result.multi_hand_landmarks:
            myhand = self.result.multi_hand_landmarks[handNo]
            # finding landmark positions as id and lm
            for id, lm in enumerate(myhand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id,cx,cy])
                if draw:
                    # cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
                     if id==4 or id ==8:
                        cv2.circle(img,(cx,cy),10,(255,0,255),10,cv2.FILLED)
        return lmList


def main():
    pTime = 0
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detect= HandDetector()
    # run webcam
    while True:
        success, img = cap.read()
        img = detect.findhands(img)
        lmList = detect.findPosition(img,draw=True)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
        cv2.imshow("image", img)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break


if __name__== "__main__":
    main()

