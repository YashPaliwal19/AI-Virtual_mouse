import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

#### Parameter ###
wCam, hCam = 640,480
frameR = 100 # Frame reduction
smoothening = 7
##################

# For smoothening
plocX, plocY = 0,0
clocX, clocY = 0,0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

wScr ,hScr = autopy.screen.size()
#print(wScr, hScr)

pTime = 0

detector = htm.handDetector(maxHands=1)

while True:
    success, img = cap.read()

    # Finding the hands
    img = detector.findHands(img)
    lmlist, bbox = detector.findPosition(img)

    # Getting the tip of the index and middle finger
    if len(lmlist)!=0:
        x1,y1 = lmlist[8][1:]
        x2,y2 = lmlist[12][1:]

    # Check which fingers are up
        fingers = detector.fingersUp()

    # Setting the frame reduction to prevent corner cases
        cv2.rectangle(img, (frameR, frameR), (wCam-frameR, hCam-frameR), (255,0,255), 2)

    # For moving mode we have only index finger up
        if fingers[1]==1 and fingers[2]==0:

            # For moving we have to interpolate between width and height of camera and screan
            x3 = np.interp(x1, (frameR,wCam-frameR), (0,wScr))
            y3 = np.interp(y1, (frameR,hCam-frameR), (0,hScr))

            # To smoothen the movement of the mouse and prevent it from Flickering
            clocX = plocX + (x3-plocX) / smoothening
            clocY = plocY + (y3-plocY) / smoothening

            # Move mouse
            autopy.mouse.move(wScr -clocX, clocY)
            cv2.circle(img, (x1,y1), 15, (255,0,255), cv2.FILLED)
            cv2.putText(img, f'Moving Mode', (450,40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 2)

            # updating the value of location
            plocX, plocY = clocX, clocY


    # For clicking mode we both middle and index finger up
        if fingers[1]==1 and fingers[2]==1:
            
            # The distance between the middle and index finger 
            length, img, lineInfo = detector.findDistance(8,12,img)
            print(length)
            cv2.putText(img, f'Click Mode', (450,40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,255), 2)
            
            if length < 35:
                cv2.circle(img ,(lineInfo[4], lineInfo[5]), 15, (0,255,0), cv2.FILLED)
            # Mouse to click
                autopy.mouse.click()
    
    # For finding frame rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20,40), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0),3)

    # Display the image
    cv2.imshow('Image', img)
    cv2.waitKey(1)