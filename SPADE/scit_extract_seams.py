import sys, getopt
import os
import cv2
import numpy as np

# python extract_blending.py ./512a.png ./512b.png ./512r.png 512

imageSize = 256

def main(args):
    DIRM = args[0]
    DIRI = args[1]
    DIRR = args[2]
    DIRC = args[3]
    size = int(args[4])

    if not os.path.isdir("./8k/map/val/"):
        try:
            os.makedirs("./8k/map/val/")
        except OSError:
            pass
    if not os.path.isdir("./8k/ins/val/"):
        try:
            os.makedirs("./8k/ins/val/")
        except OSError:
            pass
    if not os.path.isdir("./8k/real/val/"):
        try:
            os.makedirs("./8k/real/val/")
        except OSError:
            pass
    if not os.path.isdir("./8k/s/val/"):
        try:
            os.makedirs("./8k/s/val/")
        except OSError:
            pass
    if not os.path.isdir("./8k/ns/val/"):
        try:
            os.makedirs("./8k/ns/val/")
        except OSError:
            pass

    serial = 0
    imgM = cv2.imread(DIRM)
    # imgA = cv2.resize(imgA, (size, size),)
    imgI = cv2.imread(DIRI)
    imgR = cv2.imread(DIRR)
    imgC = cv2.imread(DIRC)
    if imgM.shape[0] != size:
        indice = int((imgM.shape[0] - size)/2)
        imgM = imgM[indice: indice+size, indice: indice+size, :]
        imgI = imgI[indice: indice+size, indice: indice+size, :]
        imgR = imgR[indice: indice+size, indice: indice+size, :]

    # print(img)
    m = int(size / imageSize)
    # print(m)
    for j in range(0, m-1):
        for k in range(0, m-1):
            newImgM = imgM[j * imageSize + 128: j * imageSize + 384, k * imageSize + 128: k * imageSize + 384, :]
            cv2.imwrite("./8k/map/val/" + str(serial).zfill(5) + ".png", newImgM)
            newImgI = imgI[j * imageSize + 128: j * imageSize + 384, k * imageSize + 128: k * imageSize + 384, :]
            cv2.imwrite("./8k/ins/val/" + str(serial).zfill(5) + ".png", newImgI)
            newImgR = imgR[j * imageSize + 128: j * imageSize + 384, k * imageSize + 128: k * imageSize + 384, :]
            cv2.imwrite("./8k/real/val/" + str(serial).zfill(5) + ".png", newImgR)
            newImgC = imgC[j * imageSize + 128: j * imageSize + 384, k * imageSize + 128: k * imageSize + 384, :]
            cv2.imwrite("./8k/s/val/" + str(serial).zfill(5) + ".png", newImgC)
            serial = serial + 1
            
    return 0

if __name__ == "__main__":
    # print(len(sys.argv))
    main(sys.argv[1:])