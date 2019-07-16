"""
we are labelling the downloaded raw images and saving classified image
"""

import argparse
import os
from imutils import paths
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-p","--path",required=True,help="path to images")
ap.add_argument("-o","--output",required=True,help="Path to save the labelled images")
args=vars(ap.parse_args())

imagePaths = list(paths.list_images(args["path"]))
counts={}

for(i,imagePath) in enumerate(imagePaths):
    
    print("\033[92m [INFO..] processing the image\033[00m : {}/{} ".format(i+1,len(imagePaths)))
    
    #reading image from the path
    image=cv2.imread(imagePath)
    
    #converting the image into grayscale image 'cause processing is easy in GRAYSCALE
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    #applying padding beacuse if the images in captcha attached to borders then there is a chance to loose the number in the image so we are applying padding across the border
    gray=cv2.copyMakeBorder(gray,8,8,8,8,cv2.BORDER_REPLICATE)
    
    """
    we will find the content in the raw image using blobs 
    blobs in raw image are not clear so first we have to use thresholding to make blobs clear(i.e bold)
    we can find the blobs using findContours()

    threshold() will convert foreground content as white and background as black(BINARY_INV)
    threshold the image to reveal the digits
    """
    
    thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
    
    #we will use findContours() to detect continuous blobs of pixels that are of same colour
    blobs=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #if two digits are in continuous regions of pixels then we end up making those two digits as one,
    #if we don't handle it ,we will end up in making the bad training data
    #so if there is a noise we are handling it by sorting the Contours by their area
    
    blobs = blobs[1] if imutils.is_cv2 else blobs[0]
    blobs=sorted(blobs,key=cv2.contourArea,reverse=True)[:4]
    for b in blobs:
        try:
            (x,y,w,h)=cv2.boundingRect(b)
            #region of interest will be
            roi = gray[y-5 : y+h+5, x-5 : x+w+5]

            cv2.imshow("roi",imutils.resize(roi,width=28))
            #careful when clicking the key beacuse it will be the class for the image
            key=cv2.waitKey(0)
            #if key clicked is ` then escape that image
            if(key==ord("`")):
                print("\033[93m [INFO...] ignoring the image \033[00m")
                continue



            key=chr(key).upper()
            dirPath = os.path.sep.join([args["output"],key])
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)

            count = counts.get(key,1)
            #Str().zfill() Returns a copy of the string with '0' characters padded to the leftside of the given string.
            #ex zfill(3) ==> 000name
            p=os.path.sep.join([dirPath,"{}.png".format(str(count).zfill(6))])
            cv2.imwrite(p,roi)

            counts[key]=count+1
        except KeyboardInterrupt:
            print("\033[91m Exiting Manually \033[00m")
            break
        except:
            #unknown error occured when labelling
            print("\033[91m skipping image....  \033[00m")

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    