from keras.preprocessing.image import img_to_array
from keras.models import load_model

from imutils import contours
from imutils import *
from ImageUtils.utils.captchahelper import preprocess
import numpy as np
import argparse
import cv2

ap=argparse.ArgumentParser()
ap.add_argument("-m","--model",help="path to the model",default="output/lenet.hdf5",required=False)
ap.add_argument("-i","--input",help="path to input image",required=True)
args=vars(ap.parse_args())

model=load_model(args["model"])
image=cv2.imread(args["input"])
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray=cv2.copyMakeBorder(gray,20,20,20,20,cv2.BORDER_REPLICATE)
thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
conts=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#print("hlij "+str(conts))
conts = conts[0] if is_cv2() else conts[1]
conts=sorted(conts,key=cv2.contourArea,reverse=True)
conts=contours.sort_contours(conts)[0]
output=cv2.merge([gray]*3)
predictions=[]

for c in conts:
    (x,y,w,h)=cv2.boundingRect(c)
    print("\033[90m X {} Width {} y {} height {}\033[00m".format(x,w,y,h))
    if h-w < 5:
        half_width=int(w/2)
        roi=gray[y-5:y+h+5 , x-5:x+half_width+5]
        roi=preprocess(roi,28,28)
        roi=np.expand_dims(img_to_array(roi),axis=0)/255.0
        pred=model.predict(roi).argmax(axis=1)[0]+1
        predictions.append(str(pred))

        cv2.rectangle(output,(x-2,y-2),(x+half_width+4,y+h+4),(0,255,0),2)
        cv2.putText(output,str(pred),(x-5,y-5),cv2.FONT_ITALIC,0.55,(0,255,0),2)
        print("hi hello")
        (x,y,w,h)=(x+half_width,y,half_width,h)

    roi=gray[y-5:y+h+5 , x-5:x+w+5]
    roi=preprocess(roi,28,28)
    roi=np.expand_dims(img_to_array(roi),axis=0)/255.0
    pred=model.predict(roi).argmax(axis=1)[0]+1
    predictions.append(str(pred))
    
    cv2.rectangle(output,(x-2,y-2),(x+w+4,y+h+4),(0,255,0),2)
    cv2.putText(output,str(pred),(x-5,y-5),cv2.FONT_ITALIC,0.55,(0,255,0),2)
    
print("[INFO...] \033[92m Captch predicted \033[00m : \033[91m {} \033[00m ".format("".join(predictions)))
cv2.imshow("output",output)
cv2.waitKey()
cv2.imwrite("output/"+"".join(predictions)+".jpg",output)

    
    
    