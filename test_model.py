from keras.preprocessing.image import img_to_array
from keras.models import load_model
from ImageUtils.utils.captchahelper import preprocess

import argparse
import cv2
import numpy as np

from imutils import paths
from imutils import contours
import imutils

ap=argparse.ArgumentParser()
ap.add_argument("-m","--model",required=True,help="path to load the model")
ap.add_argument("-li","--list-images",required=True,help="path to images that are to be classified")
args=vars(ap.parse_args())


"""
Loading the pretrained model LeNet
"""
print("[INFO...] Loading the model ")
model=load_model(args["model"])

"""
Loading the images from the disk and selecting some random 10 images
"""
ImagePaths=list(paths.list_images(args["list_images"]))
ImagePaths=np.random.choice(ImagePaths, size=(10,),replace=False)

for (i,imagePath) in enumerate(ImagePaths):
    image=cv2.imread(imagePath)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray=cv2.copyMakeBorder(gray,20,20,20,20,cv2.BORDER_REPLICATE)
    thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
    #find contours in the image ,keeping only the four largest ones
    """
    findContours() returns a list of (x,y) coordinates that specify the outline of each digit 
    """
    conts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    conts = conts[0] if imutils.is_cv2() else conts[1] #if cv3 then change conts[1] if imutils.is_cv3() else conts[0]
    conts=sorted(conts,key=cv2.contourArea,reverse=True)#[:4]
    conts=contours.sort_contours(conts)[0] 
    """
    it will sort according to the images coordinates because above we sorted contours according to their areas then they will 
    be orderless in the image then how will we know the exact order of the captcha in the image so we apply
    sort_contours to sort according to their coordinates(boundingBoxes) in captcha image
    for explanation: http://pyimg.co/sbm9p
    """
    #it takes our gray image and converts it into three channel images by replicating grayscale channel 3 times(1 for each R,G,B)
    output=cv2.merge([gray]*3)
    predictions=[]
    
    for c in conts:
        """
        Computing bounding box for each contour and extracting digits 
        """
        (x,y,w,h)=cv2.boundingRect(c)
        if w / h > 1.12:
            # This contour is too wide to be a single letter!
            # Split it in half into two letter regions!
            half_width = int(w / 2)
            (x,y,w,h)=(x, y, half_width, h)
            roi = gray[y-5 : y+h+5 ,x-5 : x+w+5]
            roi=preprocess(roi,28,28)
            roi=np.expand_dims(img_to_array(roi),axis=0)/255.0
            pred = model.predict(roi).argmax(axis=1)[0]+1
            predictions.append(str(pred))
            cv2.rectangle(output,(x-2,y-2),(x+w+4,y+h+4),(0,255,0),1)
            cv2.putText(output,str(pred),(x-5,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,0),2)
            (x,y,w,h)=(x + half_width, y, half_width, h)
            
        roi = gray[y-5 : y+h+5 ,x-5 : x+w+5]
        roi=preprocess(roi,28,28)
        roi=np.expand_dims(img_to_array(roi),axis=0)/255.0
        pred = model.predict(roi).argmax(axis=1)[0]+1
        predictions.append(str(pred))
        
        cv2.rectangle(output,(x-2,y-2),(x+w+4,y+h+4),(0,255,0),1)
        cv2.putText(output,str(pred),(x-5,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,0),2)
        
    print("\033[92m  [INFO...] captcha\033[00m :\033[91m {} \033[00m ".format("".join(predictions)))
    cv2.imshow("Output",output)
    cv2.waitKey()
    #I am saving the images in output folder
    cv2.imwrite("output/pred/"+str("".join(predictions))+".jpg",output)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    