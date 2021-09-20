import sys, getopt
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import cv2

guidanceSize = 64
imageSize = 256

def main(args):
    DIRF = args[0]
    DIRI = args[1]
    DIRM = args[2]
    DIRR = args[3]
    start = int(args[4])
    end = int(args[5])

    if not os.path.isdir("./8k"):
        os.mkdir("./8k")
    if not os.path.isdir("./8k/real"):
        os.mkdir("./8k/real")
    if not os.path.isdir("./8k/map"):
        os.mkdir("./8k/map")
    if not os.path.isdir("./8k/ins"):
        os.mkdir("./8k/ins")
    if not os.path.isdir("./8k/guidance"):
        os.mkdir("./8k/guidance")

    serial = 0

    imgF = Image.open(DIRF)
    size = imgF.size[0]
    imgI = Image.open(DIRI)
    size1 = imgI.size[0]
    imgI = imgI.crop((start, start, end, end))
    size1 = imgI.size[0]
    imgM = Image.open(DIRM)
    imgM = imgM.crop((start, start, end, end))
    imgR = Image.open(DIRR)
    imgR = imgR.crop((start, start, end, end))

    m = int(size / guidanceSize)

    for j in range(0, m):
        for k in range(0, m):
            newImgF = imgF.crop((k * guidanceSize, j * guidanceSize, k * guidanceSize + 64, j * guidanceSize + 64))
            # newImgA = newImgA.resize((256, 256), Image.LANCZOS)
            # newImgF = newImgF.resize((256, 256), Image.BICUBIC)
            newImgF.save("./8k/guidance/" + str(serial).zfill(5) + ".png")
            serial = serial + 1

    serial = 0
    m = int(size1 / imageSize)
    # print(m)
    for j in range(0, m):
        for k in range(0, m):
            newImgI = imgI.crop((k * imageSize, j * imageSize, k * imageSize + 256, j * imageSize + 256))
            newImgM = imgM.crop((k * imageSize, j * imageSize, k * imageSize + 256, j * imageSize + 256))
            newImgR = imgR.crop((k * imageSize, j * imageSize, k * imageSize + 256, j * imageSize + 256))
            newImgI.save("./8k/ins/" + str(serial).zfill(5) + ".png")
            newImgM.save("./8k/map/" + str(serial).zfill(5) + ".png")
            newImgR.save("./8k/real/" + str(serial).zfill(5) + ".png")
            serial = serial + 1
            
    return 0

if __name__ == "__main__":
    # print(len(sys.argv))
    main(sys.argv[1:])