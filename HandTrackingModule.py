from cv2 import FILLED
import mediapipe as mp
import cv2
import time
import math

# We are going to create a class
class handDetector():
    # below is initialization
    def __init__(self, mode=False, maxHands=2, complexity =1, detectionCon=0.5, trackCon= 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.complexity = complexity

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        # These are ids of the tip of each finger
        self.tipIds = [4,8,12,16,20]

    # Method for finding hands below
    def findHands(self, img, draw=True):

        imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handsLms in self.results.multi_hand_landmarks:
               if draw:
                    self.mpDraw.draw_landmarks(img, handsLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmlist = []
        
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            
            for id, lm in enumerate(myHand.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                xList.append(cx)
                yList.append(cy)
                #print(id,cx,cy)
                self.lmlist.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx,cy), 3, (255,0,0), cv2.FILLED)

            xmin , xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList) 
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin-20, ymin-20), (xmax+20, ymax+20), (0,255,0), 2)
        
        return self.lmlist, bbox

    # To check if the finger is up or not 
    def fingersUp(self):
        fingers=[]
        #for Thumb 
        if self.lmlist[self.tipIds[0]][1] > self.lmlist[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # For Fingers
        for id in range(1,5):
            if self.lmlist[self.tipIds[id]][2]< self.lmlist[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers

    # Method to find distance between any 2 finger points
    def findDistance(self, p1 , p2, img, draw=True, r=15, t=3):
        x1,y1 = self.lmlist[p1][1:]
        x2,y2 = self.lmlist[p2][1:]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        if draw:
            cv2.line(img, (x1,y1), (x2,y2), (255,0,255), t)
            cv2.circle(img, (x1,y1), r, (255,0,255), cv2.FILLED)
            cv2.circle(img, (x2,y2), r, (255,0,255), cv2.FILLED)
            cv2.circle(img, (cx,cy), r, (255,0,255), cv2.FILLED)
            length = math.hypot(x2-x1, y2-y1)

        return length, img, [x1,y1,x2,y2,cx,cy]

def main():
    pTime = 0
    cTime = 0 
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        sucess, img = cap.read()
        # draw=False arg in below 2 lines for not showing anything
        # Then there will be no lines 
        img = detector.findHands(img)
        lmlist, bbox = detector.findPosition(img)
        if len(lmlist)!=0:
            #print(lmlist[2])

            fingers = detector.fingersUp()
            print(fingers)

    #For frame rates
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)),(18,78), cv2.FONT_HERSHEY_PLAIN, 3,(255,255,0), 3)


        cv2.imshow("Image", img)
        cv2.waitKey(1) 


if __name__=="__main__":
    main()