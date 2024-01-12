import cv2
import mediapipe as mp 
import numpy as np 
mp_drawing = mp.solutions.drawing_utils #gives drawing utilities
mp_pose = mp.solutions.pose #imports pose estimation model from mp 

#function to find angle
def calculate_angle(a,b,c): 
    a = np.array(a) #first
    b = np.array(b) #mid
    c = np.array(c) #end

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0]) #caclulates radians
    angle = np.abs(radians*180.0/np.pi) #calculates angle

    if angle > 180.0: #max 180 degrees
        angle = 360-angle
    
    return angle 

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

        #Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            #get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y] #grabs x and y value
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y] #grabs x and y value
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y] #grabs x and y value

            #calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)

            #visualize angle
            cv2.putText(image, str(angle),
                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                       )
            
        except:
            pass

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

