# Python script to compress all .png files using "convert" in osx.

import os
import glob

imageList = []

for image in glob.glob("*.png"):
    imageName = image.replace(".png", "")
    # imageList = imageList.replace(".png", "")
    os.system("convert %s.png -quality 35 %s_compressed.jpeg" % (imageName, imageName))

