import cv2
import mediapipe as mp 
import numpy as np 
mp_drawing = mp.solutions.drawing_utils #gives drawing utilities
mp_pose = mp.solutions.pose #imports pose estimation model from mp 

cap = cv2.VideoCapture(0) #set video capture device
#setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened(): #loop through feed
        ret, frame = cap.read() #reads current feed from webcam
        
        #recolor image to put in format of RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
    
        #makes detection
        results = pose.process(image)

        image.flags.writeable = True #set writable status to true
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #recolor back to BGR

        #render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 ) #draws points on body 

        cv2.imshow('Mediapipe Feed', image) #pops up on screen

        if cv2.waitKey(10) & 0xFF == ord('q'): #checking if we hit q or close screen, to break loop
            break

    cap.release()
    cv2.destroyAllWindows()

