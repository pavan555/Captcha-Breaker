
# Captcha Breaker
I have done a project on deep learning called simple captcha breaker based on deep learning
for opencv written by Adrian Rosebrock.
 
 I am going to explain how i did this project.Here i created my own custom dataset by extracting each digits in the picture.
 
 There are mainly 4 parts in this Project.
 
 
 1. Downloading a set of captcha images
 2. Labeling and annotating images for our training
 3. Training a CNN on our custom dataset
 4. Evaluating and Testing our trained Convolutional Neural Network    

## 1. Downloading a set of captcha images
first of all, select a website that will provide captchas like the below images. I have downloaded the images from [Simple CAPTCHA](https://www.e-zpassny.com/vector/jcaptcha.do) (note: which is not working right now)


  (i)![image](https://github.com/pavan555/Captcha-Breaker/blob/master/downloads/00036.jpg?raw=true) 
  
  (ii)![image2](https://github.com/pavan555/Captcha-Breaker/blob/master/downloads/00087.jpg?raw=true)
  
  Figures : **Example Images**
  

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
