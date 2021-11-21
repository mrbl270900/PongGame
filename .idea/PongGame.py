import pygame

from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2
# Load the model
model = load_model('keras_model.h5')
cap = cv2.VideoCapture()
# The device number might be 0 or 1 depending on the device and the webcam
cap.open(0, cv2.CAP_DSHOW)
# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# Replace this with the path to your image
while(True):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
image = Image.open(frame)
#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

#turn the image into a numpy array
image_array = np.asarray(image)
# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
# Load the image into the array
data[0] = normalized_image_array

# run the inference
prediction = model.predict(data)
print(prediction)


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
cap.release()
cv2.destroyAllWindows()