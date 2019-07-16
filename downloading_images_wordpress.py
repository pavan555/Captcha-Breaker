import requests
import re
import time
import os
import argparse
import json

url = "https://contactform7.com/captcha/"
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.1.1 Safari/605.1.15',
           'Content-Type': "multipart/form-data; boundary=----WebKitFormBoundaryQgctpYC5kRiIjznW","Connection": "keep-alive",
           "Cookie": "lang = en_US;_ga = GA1.2.765999315.1562601614;_gid = GA1.2.1701684676.1562913704;__cfduid = d695b369369d5130db03260060ed2edec1562601611"
}

ap=argparse.ArgumentParser()
ap.add_argument("-o","--output",required=True,help="Path to save the images")
ap.add_argument("-n","--number",required=False,default=500,help="number of images to download")
args=vars(ap.parse_args())
s=requests.Session()
result = s.get(url, headers=headers).content.decode("UTF-8")

count =1
result = re.findall("src=\"(.*[0-9]{1,}\.png)\"", result)
for j in result:
    print("\033[095m Downloading image \033[00m : \033[092m {}/{} \033[00m  ".format(count, args["number"]))
    print(j.encode("ascii"))
    r = s.get(j.encode("ascii"), headers=headers)
    p = os.path.sep.join([args["output"], "{}.jpg".format(str(count).zfill(5))])
    f = open(p, "wb")
    f.write(r.content)
    f.close()
    time.sleep(0.1)
    count += 1

url = "https://contactform7.com/wp-json/contact-form-7/v1/contact-forms/1209/feedback"
images=["captcha-118","captcha-170","captcha-778"]
while count<args["number"]:
    try:
        s = requests.Session()
        result = json.loads(s.post(url, headers=headers).content.decode("UTF-8"))
        #print(result["captcha"])
        #print(result["captcha"][u'captcha-118'].encode("ascii"))

        for j in range(3):
            print("\033[095m Downloading image \033[00m : \033[092m {}/{} \033[00m  ".format(count,args["number"]))
            # print(j.encode("ascii"))
            r = s.get(result["captcha"][images[j]].encode("ascii"), headers=headers)
            p= os.path.sep.join([args["output"],"{}.jpg".format(str(count).zfill(5))])
            f=open(p,"wb")
            f.write(r.content)
            f.close()
            time.sleep(0.1)
            count+=1

    except Exception:
        print("\033[92m Error Downloading Webpage \033[00m")
    time.sleep(1)

