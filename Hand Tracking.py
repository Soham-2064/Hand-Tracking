import cv2
import mediapipe as mp
import time

cap= cv2.VideoCapture(0)

mphands = mp.solutions.hands #just to do these steps to use the library
hands = mphands.Hands() #these are like def of multiple hands
mpDraw = mp.solutions.drawing_utils #it will detect and visually draw the hands

#frame rates
ptime = 0
ctime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)  #process the frame for us and give the result
    # print(results.multi_hand_landmarks) #print the hands detection

    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id ,lm in enumerate(handlms.landmark):  #ratio to give the pixel value
                # print(id, lm) 
                h, w, c= img.shape       #h height, width w , c is channel of our image
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy) 

                if id== 4 :
                    cv2.circle(img, (cx, cy), 15, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handlms, mphands.HAND_CONNECTIONS)  #handlms is the single hands, and handconnections just show the connected

#for the frame rates
    ctime = time.time()
    fps = 1/ (ctime - ptime)
    ptime = ctime


    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255),3)
#---- frame rates display 

    cv2.imshow("Image", img) #to read and start the camera
    cv2.waitKey(1) #if its 0 then it would take a image

