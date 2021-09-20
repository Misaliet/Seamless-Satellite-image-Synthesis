import sys
import os
import cv2
import numpy as np
from PIL import Image


def cut(args):
    path0 = args[0]
    path1 = args[1]
    img = cv2.imread(path0)
    w, _, _ = img.shape
    indice = int(w/4)
    img = img[indice:indice*3, indice:indice*3, :]
    cv2.imwrite(path1, img)

def resize(args):
    path0 = args[0]
    path1 = args[1]
    size = int(args[2])
    images = [name for name in os.listdir(path0) if os.path.isfile(os.path.join(path0, name)) and ".jpg" or '.png' in name]

    for i in images:
        fileName = os.path.join(path0, i)
        print(fileName)
        img = Image.open(fileName)
        img = img.resize((size, size), Image.LANCZOS)
        img.save(os.path.join(path1, i))

def remove_edges(args):
    path0 = args[0]
    path1 = args[1]
    img = cv2.imread(path0)
    w, _, _ = img.shape
    indice = int(w/8)
    print(indice)
    img = img[indice:indice*7, indice:indice*7, :]
    cv2.imwrite(path1, img)

def get_final(args):
    path0 = args[0]
    path1 = args[1]
    img = cv2.imread(path0)
    w, _, _ = img.shape
    indice = int(w/6)
    img = img[indice:indice*5, indice:indice*5, :]
    cv2.imwrite(path1, img)


def main(args):
    if args[0] == 'cut':
        cut(args[1:])
    elif args[0] == 'resize':
        resize(args[1:])
    elif args[0] == 'remove_edges':
        remove_edges(args[1:])
    elif args[0] == 'get_final':
        get_final(args[1:])
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])