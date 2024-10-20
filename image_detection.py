import cv2
import numpy as np

'Color Detection Code: code written ourselves'
while True:
    #img = cv2.imread('m_and_ms.jpg', 0) #load the image in grayscale
    img = cv2.imread('m_and_ms.jpg', 1) #load the image with alpha channel
    #img = cv2.imread('m_and_ms.jpg', -1) #load the colored image
    
    #resize the image
    img = cv2.resize(img, (500,500))
    
    #convert the BGR image to Hue-Saturation-Value
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    #define your color bounds that you want to isolate
    lower_bound = np.array([110,50,50])
    upper_bound = np.array([130,255,255])
    
    #create a mask from the img_hsv based on the bounds
    mask = cv2.inRange(img_hsv, lower_bound, upper_bound)
    
    img_iso = cv2.bitwise_and(img, img, mask = mask)
    
    # show the image, mask, and final image in a new window.
    #first parameter is the window label, and the second parameter is the image
    cv2.imshow('image', img)
    cv2.imshow('image mask', mask)
    cv2.imshow('isolated image', img_iso)
    
    ##indicate how long you want to keep the window open
    #use 0 if you want to keep it open till you press any key.
    #press q to exit the while loop
    key = cv2.waitKey(1) & 0xFF

    # Break the loop if 'q' is pressed or windows are closed
    if key == ord('q'):
        break

    # Check if any of the windows were closed
    if cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) < 1:
        cv2.destroyWindow('image')
    if cv2.getWindowProperty('image mask', cv2.WND_PROP_VISIBLE) < 1:
        cv2.destroyWindow('image mask')
    if cv2.getWindowProperty('isolated image', cv2.WND_PROP_VISIBLE) < 1:
        cv2.destroyWindow('isolated image')

#Make sure to destroy all opened windows when you are done
cv2.destroyAllWindows()


'''
Face Detection Code: based off code from _______________ modified to label face and eye regions as well as quit with our parameters
'''

# cap = cv2.VideoCapture(0)
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# #continuously loop so that you keep updating the frame
# while True:
#     #look at the frame from the webcam - ret is a boolean stating true if frame is returned
#     ret, frame = cap.read()
    
#     #convert the frame into a grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     #find all the faces in the frame using the haar cascades algorithm
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     #loop through all the faces which are rectangles
#     for (x,y,w,h) in faces:
#         #x and y are the starting point of the rect, w and h are width/height
        
#         #draw the rectangle based on the retrieved rectangels
#         cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 3)
#         cv2.putText(frame, 'face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 0, 0), 2, cv2.LINE_AA)
#         #find the area in the image that represent the face in the gray and frame
#         roi_gray = gray[y:y+w, x:x+w]
#         #this references the frame image directly
#         roi_color = frame[y:y+h, x:x+w]
#         eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
#         for (ex,ey,ew,eh) in eyes:
#             #draw the rectangle around the eyes in the roi_color
#             #remember you want to reference the roi_color not the roi_gray since
#             #roi_color refereces the actual frame while roi_gray does not.
#             cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 3)
#             cv2.putText(roi_color, 'eyes', (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
#     #after all the changes, show the image with the drawings
#     cv2.imshow('frame', frame)
    
#     #press q to exit the while loop
#     if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
#         break
    
# #make sure to release the captured video and destroy the windows.
# cap.release()
# cv2.destroyAllWindows()