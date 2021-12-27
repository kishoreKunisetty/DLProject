import streamlit as st
import cv2
import mediapipe as mp
import pickle
import numpy as np
from time import sleep

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# load model
file_name = "RF_model_full.pkl"
model = pickle.load(open(file_name,'rb'))

classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D','E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q','R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale = 4
fontColor = (255,0,0)
thickness = 4
lineType = 2

st.title("DL Project")
run = st.checkbox('run')
FRAME_WINDOW = st.image([])

def print_points(lms,size):
    w, h, d = size
    for i in [8,12,16]:
        x  = lms.landmark[i].x
        y = lms.landmark[i].y
        z = lms.landmark[i].z
        st.write(f"x {x*size[0]} y {y*size[1]} z {z*size[0]} ")
        # print(lms.landmark[i], type(lms.landmark[i]),lms.landmark.x)
    # for id, lm in enumerate(lms.landmark):
    #     px, py = int(lm.x * w), int(lm.y * h)
    # print(lm.landmark[8])
    # print(px)
    return 

def clean(dic):
    len_dic = len(dic.keys())
    features = []
    temp = [0]*42
    for side, item in dic.items():
        for lm in item.landmark:
            features.append(lm.x)
            features.append(lm.y)
    if len(features) == 42 and list(dic.keys())[0] == 'Right':
        features = features + temp 
    if len(features) == 42 and list(dic.keys())[0] == 'Left':
        features = temp + features
#     print(features, len(features))
    return features

cap = cv2.VideoCapture(0)
while run:
    with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
        while run:
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            size = image.shape
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.multi_hand_landmarks:
                land_marks = {}
                for hand_type, hand_landmarks in zip(results.multi_handedness , results.multi_hand_landmarks):
                    # print(f"[INFO] {i} {hand_landmarks}")
                    # print_points(hand_landmarks,size)
                    # model needed.
                    land_marks[hand_type.classification[0].label] = hand_landmarks
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            # Flip the image horizontally for a selfie-view display.
                x = np.array(clean(land_marks))
                # print(x)
                x_ = x.reshape(1,-1)
                # print("[INFO] X : ",x," X_ ",x_)
                y = model.predict(x_)
                # print(type(y), y[0])
                cv2.putText(image,classes[y[0]], bottomLeftCornerOfText, font, fontScale,fontColor,thickness,lineType)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            # cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            # FRAME_WINDOW.image(cv2.flip(image, 1))
            FRAME_WINDOW.image(image)
            # if cv2.waitKey(5) & 0xFF == 27:
            #     break
            # sleep(2)
    
    cap.release()
else : 
    st.write("Stopped")
