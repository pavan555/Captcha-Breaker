# Captcha Breaker
I have done a project on deep learning called simple captcha breaker based on deep learning
for opencv written by Adrian Rosebrock.
 
 I am going to explain how i did this project.Here i created my own custom dataset by extracting each digits in the picture.
 
 
 ![example](https://github.com/pavan555/Captcha-Breaker/blob/master/downloads/00446.jpg?raw=true) ==> 1738

 
 There are mainly 4 parts in this Project.
 
 
 1. Downloading a set of captcha images
 2. Labeling and annotating images for our training
 3. Training a CNN on our custom dataset
 4. Evaluating and Testing our pre-trained LeNet model  


## 1. Downloading a set of captcha images
first of all, select a website that will provide captchas like the below images. I have downloaded the images from [Simple CAPTCHA](https://www.e-zpassny.com/vector/jcaptcha.do) (note: which is not working right now)


 
   ![ex images](https://user-images.githubusercontent.com/25476729/63042877-a1ef1080-bee8-11e9-984f-6b4b5d5f4c23.png)

  Figure : **Example Images**
  
### The sample code for downloading the images is displayed below.

```python


r= requests.get("https://www.e-zpassny.com/vector/jcaptcha.do")
path=os.path.sep.join([args["output"],"{}.jpg".format(str(total).zfill(5))])
file = open(path,"w")
file.write(r.content)
file.close()


```
I downloaded approximately 500 images and saved in downloads path.

However, these are just raw captcha images.
We need to extract and label each of the digits in the captchas to create our training data set.
This will be accomplished by next section.




##  2. Labeling and annotating images for our training

To extract each digit and label our digit, i used [label_the_data.py](https://github.com/pavan555/Captcha-Breaker/blob/63851759c537364f52c958dc643f30f72b2e66a1/label_the_data.py) file.

OpenCV is a popular framework for computer vision and image processing. I utilized OpenCV for processing the captcha images.

The basic idea is to split the CAPTCHA image into four identical parts( because we know that there will be only 4 digits in the image) and label the data.
The main problem here is splitting 

(i) we can't do it manually 'cause it would take months to split and label each image

(ii) we can't just split the image into four equal-sized chunks because the CAPTCHA randomly places the digits in different horizontal positions.
like below

![gif](https://user-images.githubusercontent.com/25476729/63044371-b08af700-beeb-11e9-858b-89738961326a.gif)

Luckily,We can automate these using OpenCV...

(i) So we will start with raw CAPTCHA image

![captcha](https://user-images.githubusercontent.com/25476729/63083566-a8bc6880-bf66-11e9-82cc-9609c2fd8929.png)


(ii) Use padding beacuse if the images in captcha attached to borders then there is a chance to loose the number in the image so we are using padding across the border.

[lines]:https://github.com/pavan555/Captcha-Breaker/blob/63851759c537364f52c958dc643f30f72b2e66a1/label_the_data.py#L19-L30


(iii) Then we will convert our image into pure white and black (this is called thresholding)
so that it will be easier to find continuous regions.
![Threshold](https://user-images.githubusercontent.com/25476729/63087231-3dc35f80-bf6f-11e9-9409-35ab8cd5ba97.png)

(iv) we will use OpenCV's _**findContours()**_  function to detect the digits that contain continuous blobs of pixels together.
![digits](https://user-images.githubusercontent.com/25476729/63091055-cdbad680-bf7a-11e9-8086-258abd3f6297.png)

(v) Then each rectangle( Region of interest )  will be displayed to classify with hand and we can save the images in the corresponding classified directory according to the entered key by us.
Those ROI images will be given like below.
![roi1](https://user-images.githubusercontent.com/25476729/63092132-4d967000-bf7e-11e9-9224-c2d8192f75bc.png)




Now that we have a way to extract individual digits,run it across all CAPTCHA images we have.The Goal is to collect
different variations of a each digit.we can save each digit in it's own folder to keep things organized.

Here's picture of my 6 digit dataset after i extracted all of my captcha images.

![6-digit dataset](https://user-images.githubusercontent.com/25476729/63092946-08c00880-bf81-11e9-82ed-d0c9e22c0bc2.png)

Figure : *Some of the 6 digits extracted from 2000+ images. I ended up with 254 different "6" images*


### The sample code for annotating the images is displayed below.



```python

thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
blobs=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
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
```


## 3. Training a CNN on our custom dataset
Since we need to recognize only Single-digit, So we don't need a complex neural network architecture.
Recognizing digits is easier than recognizing complex images like a cat,dog...etc


![process](https://user-images.githubusercontent.com/25476729/63113613-83ece300-bfb0-11e9-837e-fdd163965730.png)

* Defining the Simpler Convolution Neural Network LeNet( It is small and easy to understand — yet large enough to provide interesting results)
to recognize the digits.

* Defining this LeNet architecture takes only few lines of code using Keras.



```python
##LeNet Architecuture is 
# CONV==>TANH==>POOL==>CONV==>TANH==>POOL==>FC==>TANH==>FC==>SOFTMAX

model = Sequential()
model.add(Conv2D(20,(5,5),padding="same",input_shape=image))

model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(50,(3,3),padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))
model.add(Dense(classes))
model.add(Activation("softmax"))

```

* Now we can train the model using our custom dataset.


```python
model.fit(trainX,trainY,validation_data=(testX,testY),batch_size=32,epochs=15,verbose=1)
```
* After nearly 10 passes, we hit 100% accuracy, and at last, we can bypass the captcha images.

Save this Trained model for further use.
```python 

model.save("output/lenet.hdf5")

```

##  4. Evaluating our pre-trained model
Now that we have a trained neural network,using it to break Digit CAPTCHA's is very easy.

1. Grab a Real World digit CAPTCHA image from a website like the ones I downloaded.
2. Break up the CAPTCHA image into four separate digit images using the same approach used in annotating images.
3. let our trained model predict each digit in separate images.
4. Use those predicted digits to break the captcha image into text.

Here are some of the outputs predicted by our model:

![result](https://user-images.githubusercontent.com/25476729/63114250-06c26d80-bfb2-11e9-81bc-6d5176820c9e.gif)

from the command line:

![ezgif com-video-to-gif](https://user-images.githubusercontent.com/25476729/63115113-1a6ed380-bfb4-11e9-95a9-c889d56b44a9.gif)
----__*Finished*__---- :+1: :octocat: :revolving_hearts: