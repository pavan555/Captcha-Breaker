
"""
It will be responsible for downloading image  captchas from our website 
and save it into our disk

it will store the raw captcha images to our disk

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}


"""

import requests
#the requests library will make http connections easy and heavily used in python
import argparse
import os
import time

url="https://www.e-zpassny.com/vector/jcaptcha.do"

ap=argparse.ArgumentParser()
ap.add_argument("-u","--url",help="if the images are to be downloaded from url give url",required=False,default=url)
ap.add_argument("-o","--output",required=True,help=" path to save the images")
ap.add_argument("-n","--number",type=int,default=500,help="# of images to download")
args=vars(ap.parse_args())


total=0 

for i in range(0,args["number"]):
    try:
        #requesting web page from our script
        r=requests.get(args['url'],timeout=60, headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3'})
        
        #save the image
        p = os.path.sep.join([args["output"],"{}.jpg".format(str(total).zfill(5))])
        #Str().zfill() Returns a copy of the string with '0' characters padded to the leftside of the given string.
        #ex zfill(3) ==> 000name
        f= open(p,"wb")
        f.write(r.content)
        f.close()
        
        print("[INFO..] Downloaded image {}".format(total))
        total+=1
        
    except:
        print("Error downloading image")
        
    time.sleep(0.2)
