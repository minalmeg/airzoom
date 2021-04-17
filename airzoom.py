"""
Air Zoom
Usage - python airzoom.py --p test.png
"""
 
import cv2
import mediapipe as mp
import time
import argparse

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
 
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
 
    def findHands(self, img, draw=False):       #detect hands
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
 
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img
 
    def findPosition(self, img, handNo=0, draw=True):       #find postions of each finger 
 
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw and id == 4 or id == 8:             #4 = thumb and 8 = index finger
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
 
        return lmList

    def distance_between_fingers(self, lmList):
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        x3, y3 = lmList[0][1], lmList[0][2]
        if x3 > x1 :
            flag = 1            #left
        elif x3 < x1:
            flag = 0            #right
        a,b = x1-x2,y1-y2    
        diff = abs(a-b)
        return diff,y2,x1


    def overlay_transparent(self,background_img, img_to_overlay_t, x, y):
        overlay_size = None
        
        bg_img = background_img.copy()
        
        if overlay_size is not None:
            img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

        # Extract the alpha mask of the RGBA image, convert to RGB 
        b,g,r,a = cv2.split(img_to_overlay_t)
        overlay_color = cv2.merge((b,g,r))
        
        # Apply some simple filtering to remove edge noise
        mask = cv2.medianBlur(a,5)

        h, w, _ = overlay_color.shape
        roi = bg_img[y:y+h, x:x+w]

        # Black-out the area behind the logo in our original ROI
        img1_bg = cv2.bitwise_and(roi.copy(),roi.copy(),mask = cv2.bitwise_not(mask))
        
        # Mask out the logo from the logo image.
        img2_fg = cv2.bitwise_and(overlay_color,overlay_color,mask = mask)

        # Update the original image with our new ROI
        bg_img[y:y+h, x:x+w] = cv2.add(img1_bg, img2_fg)
     

        return bg_img
     
 
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', action='store', type=str, required = True)
    args = parser.parse_args()
    s_img_path = args.p        
    s_img = cv2.imread(s_img_path,-1)
    check = s_img_path.split(".")
    for i in check:
        if i == "jpeg" or i == "jpg":
            s_img = cv2.cvtColor(s_img, cv2.COLOR_BGR2RGBA)
    orig_img = s_img
    flag = 0        #flag = 1 if left hand else right
    writer = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'MJPG'),25, (1200,720))
    save_video = True
    while True:
        success, img = cap.read()
        img = detector.findHands(img)           #detect hand
        lmList = detector.findPosition(img)     #find x,y coordinates of fingers  
        if len(lmList) != 0:
            diff, y2, x1 = detector.distance_between_fingers(lmList)        #determines differenc between index finger and thumb, x,y coordinate for png to paste
            if diff == 0 or diff == 1:      #when the distance between the fingers is 0 or 1
                continue
            else:
                try:
                    s_img = cv2.resize(s_img,(int(diff),int(diff)), interpolation = cv2.INTER_CUBIC)        #resize image so that it's equal to difference between distance index finger and thumb
                    h1, w1,_ = s_img.shape 
                    x = int(x1-w1)
                    if flag == 1:           #left hand
                        img = detector.overlay_transparent(img, s_img, x1, y2)          #paste the image
                        #img[y2:y2+h1,x1:x1+w1] = s_img
                    elif flag == 0:         #right hand
                        img = detector.overlay_transparent(img, s_img, x, y2)
                        #img[y2:y2+h1,x1-w1:x1] = s_img
                    s_img = orig_img        #restore the resized image to maintain the original resolution
                except :
                    pass
        img = cv2.resize(img, (1200,720)) #resize frame
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
 
        cv2.imshow("Image", img)
        cv2.waitKey(1)
        if save_video:
            writer.write(img)
    result.release() 
 
if __name__ == "__main__":
    main()