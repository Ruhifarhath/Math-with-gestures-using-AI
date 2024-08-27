import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
from PIL import Image
import streamlit as st

genai.configure(api_key="AIzaSyDnsszA9QGOFxMy_d2YfQ4SjElzMprryBc")
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Initialize the webcam to capture video
cap = cv2.VideoCapture(0)
cap.set(propId=3, value=1280)  # width
cap.set(propId=4, value=720)   # height

# Initialize the HandDetector class
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

def getHandInfo(img):
    hands, img = detector.findHands(img, draw=True, flipType=True)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers1 = detector.fingersUp(hand)
        return fingers1, lmList
    else:
        return None

def draw(info, prev_pos, canvas):
    if info:
        fingers, lmList = info
        current_pos = None
        if fingers == [0, 1, 0, 0, 0]:
            current_pos = lmList[8][0:2]  # Index finger tip coordinates
            if prev_pos is None:
                prev_pos = current_pos
            cv2.line(canvas, prev_pos, current_pos, (255, 0, 255), 10)  # Draw purple line
        elif fingers==[1,0,0,0,0]:
             canvas = np.zeros_like(img)

    return current_pos, canvas

#
def sendToAI(model,canvas,fingers):
  if fingers == [0,1,1,1,1]:
     pil_image = Image.fromarray(canvas)
     response = model.generate_content(["Solve this math problem", pil_image])
     print(response.text)

    
prev_pos = None
canvas = None
image_combined = None

# Continuously get frames from the webcam
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Horizontally flip the image

    if canvas is None:
        canvas = np.zeros_like(img)
        image_combined = img.copy()

    info = getHandInfo(img)
    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(info, prev_pos, canvas)
        sendToAI(model,canvas,fingers)

    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)

    # Display the images
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", canvas)
    cv2.imshow("image_combined", image_combined)

    # Wait for 1 millisecond between frames
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
