import imutils
import cv2

def preprocess(image,width,height):
    
    (h,w)=image.shape[:2]
    
    if w>h :
        #if the width is greater than the height then resize along the width
        image = imutils.resize(image,width=width)
    else:
        #else resize along the height
        image = imutils.resize(image,height=height)


    #determine padding for width and heigth to pad to get required width & height
    padW = int((width - image.shape[1])/2.0)
    padH = int((height - image.shape[0])/2.0)
    
    image = cv2.copyMakeBorder(image,padH,padH,padW,padW,cv2.BORDER_REPLICATE)
    image = cv2.resize(image,(width,height))
    #return preprocessed image
    return image

