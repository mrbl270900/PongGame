import pygame

from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

import argparse
import os
import sys
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import os
import sys
import subprocess

if sys.platform == 'linux':
    from gpiozero import CPUTemperature

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fullscreen',
                    help='Display window in full screen', action='store_true')
parser.add_argument(
    '-d', '--debug', help='Display debug info', action='store_true')
parser.add_argument(
    '-fl', '--flip', help='Flip incoming video signal', action='store_true')
args = parser.parse_args()

# create model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.load_weights('model.savedmodel')

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

def get_gpu_temp():
    temp = subprocess.check_output(['vcgencmd measure_temp | egrep -o \'[0-9]*\.[0-9]*\''],
                                   shell=True, universal_newlines=True)
    return str(float(temp))

# start the webcam feed
cap = cv2.VideoCapture(0)
while True:
    # time for fps
    start_time = time.time()

    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if args.flip:
        frame = cv2.flip(frame, -1)
    if not ret:
        break
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(
            cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        emotion_label = emotion_dict[maxindex]
        cv2.putText(frame, emotion_label, (x+20, y-60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # full screen
    if args.fullscreen:
        cv2.namedWindow("video", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("video", cv2.WND_PROP_FULLSCREEN, 1)

    # debug info
    if args.debug:
        fps = str(int(1.0 / (time.time() - start_time)))
        cv2.putText(frame, fps + " fps", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        if sys.platform == 'linux':
            cpu_temp = str(int(CPUTemperature().temperature)) + " C (CPU)"
            cv2.putText(frame, cpu_temp, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, get_gpu_temp() + " C (GPU)", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('video', cv2.resize(
        frame, (800, 480), interpolation=cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



pygame.init() #her intitere vi vores pygame samt vores fonter
pygame.font.init()
my_font = pygame.font.SysFont('Helvetica', 20)
hight = 300
width = 500
size_paddle_y = 50
size_paddle_x = 5
size_ball = 15
screen = pygame.display.set_mode([width, hight])
running = True
color = (255,255,255)
xy_c = [250,150]
speed_x = 0.1
speed_y = 0.1
count = 10


while running: #her startes vores whille loop
    screen.fill((0,0,0))
    y1 = pygame.mouse.get_pos()[1]
    if y1 > hight-size_paddle_y:
        y1 = hight-size_paddle_y
    pygame.draw.rect(screen, color, pygame.Rect(1,y1,size_paddle_x,size_paddle_y))



    if y1 > hight-size_paddle_y:
        y1 = hight-size_paddle_y
    pygame.draw.rect(screen, color, pygame.Rect(width-size_paddle_x,xy_c[1]-(size_paddle_y/2),size_paddle_x,size_paddle_y))

    xy_c[0] = xy_c[0] + speed_x
    xy_c[1] = xy_c[1] + speed_y

    if xy_c[1] > hight-size_ball or xy_c[1] < 0+size_ball:
        speed_y=speed_y*(-1)

    pygame.draw.circle(screen, color, xy_c,size_ball)

    if xy_c[0]+size_ball > width-size_paddle_x and count < 0:
        if xy_c[1]-size_ball < xy_c[1]+size_paddle_y and xy_c[1]+size_ball > xy_c[1]:
            speed_x = speed_x * (-1)
            count = 10

    if xy_c[0]-size_ball < 1+size_paddle_x and count < 0:
        if xy_c[1]-size_ball < y1+size_paddle_y and xy_c[1]+size_ball > y1:
            speed_x = speed_x * (-1)
            count = 10

    if xy_c[0]+size_ball > width+1 or xy_c[0]-size_ball < 0-1:
        xy_c[0]=1000
        lost_text_field = my_font.render("Game Over", False, color)
        play_again_text = my_font.render("Play Again by clicking your mouse", False, color)
        screen.blit(lost_text_field,(width/2 - 30,hight/2))
        screen.blit(play_again_text, (width/2 - 30,hight/2 + 25))
        for ev in pygame.event.get():
            if ev.type == pygame.MOUSEBUTTONDOWN:
                xy_c[0] = 250
                xy_c[1] = 150
                speed_x = 0.1
                speed_y = 0.1
                count = 10
    count = count - 1

    pygame.display.flip()
    for event in pygame.event.get(): # dette bruges til at stoppe pogramet når man trykker på krydset i pygame vinduet
        if event.type == pygame.QUIT:
            running = False