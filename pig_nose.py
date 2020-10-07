import cv2 
import numpy as np 
import dlib 


cap = cv2.VideoCapture(0)
pig_img = cv2.imread("piggy.png")

detector = dlib.get_frontal_face_detector() # for face detection
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:

        #----------detect landmark------
        landmarks = predictor(gray,face)
        
        top_nose = (landmarks.part(29).x, landmarks.part(29).y )
        center_nose = (landmarks.part(30).x, landmarks.part(30).y )
        left_nose = (landmarks.part(31).x, landmarks.part(31).y )
        right_nose = (landmarks.part(35).x, landmarks.part(35).y )
        
        # calculate the nose width and height, watch video for the logic and formula is (x2-x1)**2 + (y2-y2)**2
        nose_width = np.sqrt((left_nose[0] - right_nose[0])**2 + (left_nose[1] - right_nose[1])**2)

        nose_height = nose_width*0.65   # 0.65 is the ratio of width to height of the image

        # i have used my own logic to draw rectangle  here over the nose
        # cv2.rectangle(frame,(int(left_nose[0]),int(left_nose[1]-nose_height)),(int(right_nose[0]),int(right_nose[1])),(0,0,255),2)
        
        #coordinates of the nose
        
        top_left = int(left_nose[0]),int(left_nose[1]-nose_height)
        bottom_right = int(right_nose[0]),int(right_nose[1]) 

        #resize the nose as per my changing nose area
        pig_img_resized = cv2.resize(pig_img,(int(nose_width),int(nose_height)))

        # convert to grayscale and threshold the pig_nose image 
        pig_img_gray = cv2.cvtColor(pig_img_resized,cv2.COLOR_BGR2GRAY)
        _,nose_mask = cv2.threshold(pig_img_gray,240,255,cv2.THRESH_BINARY)

        # roi is taken, the part having the nose
        nose_area = frame[top_left[1]:top_left[1]+int(nose_height),top_left[0]:top_left[0]+int(nose_width)]

        # putting the mask on the nose
        nose_with_mask = cv2.bitwise_and(nose_area,nose_area,mask = nose_mask)
        # adding the pig_img_resized on the nose_with_mask
        final_nose = cv2.add(nose_with_mask,pig_img_resized)
        #(final_nose matlab nose ma pig_img lagisakyo )

        # the part in frame where nose exists must me assigned with the nose
        frame[top_left[1]:top_left[1]+int(nose_height),top_left[0]:top_left[0]+int(nose_width)] = final_nose

    try:
        cv2.imshow("frame",frame)
    except:
        pass
    # cv2.imshow("pig_img_resized",pig_img_resized)
    # # cv2.imshow("nose_with_mask",nose_with_mask)
    # cv2.imshow("final_nose",final_nose)


    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()