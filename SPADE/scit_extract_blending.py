import sys, getopt
import os
import cv2
import numpy as np

# python scit_extract_blending.py ./zraw/z1m.png ./zraw/z1i.png ./zraw/z1r.png 1024

imageSize = 256
guidanceSize = 64

def main(args):
    DIRM = args[0]
    DIRI = args[1]
    DIRR = args[2]
    size = int(args[3])
    if len(args) == 5:
        DIRC = args[4]

    if not os.path.isdir("./8k"):
        os.mkdir("./8k")
    if not os.path.isdir("./8k/real"):
        os.mkdir("./8k/real")
    if not os.path.isdir("./8k/map"):
        os.mkdir("./8k/map")
    if not os.path.isdir("./8k/ins"):
        os.mkdir("./8k/ins")
    if len(args) == 5:
        if not os.path.isdir("./8k/guidance"):
            os.mkdir("./8k/guidance")

    serial = 0
    imgM = cv2.imread(DIRM)
    # imgA = cv2.resize(imgA, (size, size),)
    imgI = cv2.imread(DIRI)
    imgR = cv2.imread(DIRR)
    if len(args) == 5:
        imgC = cv2.imread(DIRC)
        if imgM.shape[0] != size:
            indice = int((imgM.shape[0] - size)/2)
            imgM = imgM[indice: indice+size, indice: indice+size, :]
            imgI = imgI[indice: indice+size, indice: indice+size, :]
            imgR = imgR[indice: indice+size, indice: indice+size, :]

    m = int(size / imageSize)
    # print(m)
    for j in range(0, m-1):
        for k in range(0, m-1):
            newImgM = imgM[j * imageSize + 128: j * imageSize + 384, k * imageSize + 128: k * imageSize + 384, :]
            cv2.imwrite("./8k/map/" + str(serial).zfill(5) + ".png", newImgM)
            newImgI = imgI[j * imageSize + 128: j * imageSize + 384, k * imageSize + 128: k * imageSize + 384, :]
            cv2.imwrite("./8k/ins/" + str(serial).zfill(5) + ".png", newImgI)
            newImgR = imgR[j * imageSize + 128: j * imageSize + 384, k * imageSize + 128: k * imageSize + 384, :]
            cv2.imwrite("./8k/real/" + str(serial).zfill(5) + ".png", newImgR)
            if len(args) == 5:
                newImgC = imgC[j * guidanceSize + 32: j * guidanceSize + 96, k * guidanceSize + 32: k * guidanceSize + 96, :]
                cv2.imwrite("./8k/guidance/" + str(serial).zfill(5) + ".png", newImgC)
            
            serial = serial + 1
            
    return 0

if __name__ == "__main__":
    # print(len(sys.argv))
    main(sys.argv[1:])