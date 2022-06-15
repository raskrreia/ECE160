
import cv2 as cv
import numpy as np
import time
import telepot

token = '5319436243:AAEMiq5mdxECK1Yzixg3rdxGi4X_7MWP87Q' # telegram token
receiver_id = 1810976786 # https://api.telegram.org/botTOKEN/getUpdates
#   https://api.telegram.org/bot5319436243:AAEMiq5mdxECK1Yzixg3rdxGi4X_7MWP87Q/getUpdates

#   Confidence Threshold & Non-Max Suppression Threshold
Conf_threshold = 0.5
NMS_threshold = 0.7

COLORS = [(255,0,255),(0,0,255)]
"""
    Color Code:
        Red = 0,0,255 #No Helmet
        Violet = 0,255,0 #With Helmet
"""

# Activate
bot = telepot.Bot(token)
bot.sendMessage(receiver_id, 'Safety Helmet Detection is now active') # send a activation message to telegram receiver id

#   Class Names: No Mask, Surgical Mask, Fabric Mask, FFP Mask
class_name = []
with open('helmet.names','r') as f:
    class_name = [cname.strip() for cname in f.readlines()]
    
print(class_name)
print("\nSafety Helmet Detection is now active.")

#   Load YOLOv4-Tiny Model
net = cv.dnn.readNet('helmet.weights','helmet.cfg')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

#   Detect Model
model = cv.dnn_DetectionModel(net)
#   Set Parameters
model.setInputParams(size=(416,416), scale = 1/255, swapRB=True)
media_source = 'helmetest.mp4' #0 for main, 1 = secondary cam,... ('filename.mp4') for videos

#   Source Feed: Webcam
cap = cv.VideoCapture(media_source)

while True:
    ret, frame = cap.read()

    
    if ret == False:
        break
    
    classes, scores, boxes = model.detect(frame,Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip (classes, scores, boxes):
        
        classes = np.argmax(scores) 
        confidence = scores[classes]

        color = COLORS[int(classid)%len(COLORS)]
        label = "%s : %f" % (class_name[classid[0]], (score*100))+"%"
        
#   Bounding Box Design

        cv.rectangle(frame,box,color,1)
        cv.putText(frame, label, (box[0], box[1]-10), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.45,color, 2)
        
    
#   Save picture with No mask
    
        if classid== 1:
#        cv.putText(frame,'WEAR MASK!', (box[0], box[1]-30), 
 #                      cv.FONT_HERSHEY_SIMPLEX, 0.45,color, 2)
            cv.rectangle(frame,box,color,1)
            cv.putText(frame, label, (box[0], box[1]-10), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.45,color, 2)
#"""
            t = time.strftime("%Y-%m-%d_%H-%M-%S")
            print("Image "+t+" saved"+" No Helmet Detected")
        
        #   Change path according to location
            time_stamp = int(time.time())
            fcm_photo = f'Detected/{time_stamp}.png'
            cv.imwrite(fcm_photo, frame) # notification photo
            bot.sendPhoto(receiver_id, photo=open(fcm_photo, 'rb')) # send message to telegram
            print(f'{time_stamp}.png has been sent (NO HELMET DETECTED).')
            time.sleep(1) # wait for 1 second. Only when it detects.
#"""
    cv.imshow('Helmet Detection (160 Project)', frame)
    
    key = cv.waitKey(1)

#   To exit our project, Press 'Q' sa keyboard.
    if key == ord('q'):
        break
    
print("Safety Helmet Detector Closed")

cap.release()
cv.destroyAllWindows()
