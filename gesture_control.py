import cv2 as c
import numpy as np
import math
import pyautogui as p
import time as t

cap=c.VideoCapture(0)
def nothing(x):
    pass
#window name
c.namedWindow(winname="Color Adjustment")     
c.resizeWindow("Color Adjustment",(300,300))
#creating the trackbar for adjusting thresh image
c.createTrackbar("Thresh","Color Adjustment",0,255,nothing)
#color detection track
c.createTrackbar("lower-h","Color Adjustment",0,255,nothing)                    
c.createTrackbar("lower-s","Color Adjustment",0,255,nothing)
c.createTrackbar("lower-v","Color Adjustment",0,255,nothing)
c.createTrackbar("upper-h","Color Adjustment",255,255,nothing)
c.createTrackbar("upper-s","Color Adjustment",255,255,nothing)
c.createTrackbar("upper-v","Color Adjustment",255,255,nothing)
while True:
    _,frame=cap.read()
    frame=c.flip(frame,2)
    frame = c.resize(frame, (320, 240)) 
    c.rectangle(frame,(0,1),(300,500),(255,0,0),0)
    crop_image = frame[1:240, 0:160]
    
    hsv=c.cvtColor(crop_image,c.COLOR_BGR2HSV)
    #detecting hand
    l_h=c.getTrackbarPos("lower-h","Color Adjustment")
    l_s=c.getTrackbarPos("lower-s","Color Adjustment")
    l_v=c.getTrackbarPos("lower-v","Color Adjustment")
    u_h=c.getTrackbarPos("upper-h","Color Adjustment")
    u_s=c.getTrackbarPos("upper-s","Color Adjustment")
    u_v=c.getTrackbarPos("upper-v","Color Adjustment")

    lowerb=np.array([l_h,l_s,l_v])
    upperb=np.array([u_h,u_s,u_v])

    mask=c.inRange(hsv,lowerb,upperb)
    filtr=c.bitwise_and(crop_image,crop_image,mask=mask)

    #enhancing pixel value
    mask1=c.bitwise_not(mask)
    m_g=c.getTrackbarPos("Thresh","Color Adjustment")
    ret,thresh=c.threshold(mask1,m_g,255,c.THRESH_BINARY)
    dilate=c.dilate(thresh,(3,3),iterations=6)

    cnts,hier=c.findContours(thresh,c.RETR_TREE,c.CHAIN_APPROX_SIMPLE)
    try:
        cm=max(cnts,key=lambda x: c.contourArea(x))
        epsilon=0.0005*c.arcLength(cm,True)
        data=c.approxPolyDP(cm,epsilon,True)
        hull=c.convexHull(cm)

        c.drawContours(crop_image,[cm],-1,(50,50,150),2)
        c.drawContours(crop_image,[hull],-1,(0,255,0),2)

        #convexity defect
        hull=c.convexHull(cm,returnPoints=False)
        defects=c.convexityDefects(cm,hull)
        count_defect=0
        for i in range(defects.shape[0]):
            s,e,f,d=defects[i,0]

            start=tuple(cm[s][0])
            end=tuple(cm[e][0])
            far=tuple(cm[f][0])
            #cosine rule
            a=math.sqrt((end[0]-start[0])**2+(end[1]-start[1])**2)
            b=math.sqrt((far[0]-start[0])**2+(far[1]-start[1])**2)
            d=math.sqrt((end[0]-far[0])**2+(end[1]-far[1])**2)
            angle=(math.acos((b**2+d**2-a**2)/(2*b*d))*180)/3.14

            if angle<=50:
                count_defect+=1
                c.circle(crop_image,far,5,[255,255,255],-1)

            if count_defect==0:
                c.putText(frame," ",(50,50),c.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)

            elif count_defect==1:
                p.press("space")
                c.putText(frame,"play/pause",(50,50),c.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
            elif count_defect==2:
                p.press("up")
                c.putText(frame,"volume up",(50,50),c.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
            elif count_defect==3:
                p.press("down")
                c.putText(frame,"volume down",(50,50),c.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
            elif count_defect==4:
                p.press("right")
                c.putText(frame,"Forward",(50,50),c.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
            else:
                pass
    except:
        pass
    c.imshow("thresh",thresh)
    c.imshow("filter",filtr)
    c.imshow("result",frame)
    if c.waitKey(1)==ord("s"):
        break
cap.release()
c.destroyAllWindows()