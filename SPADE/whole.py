import sys
import os
import cv2
import numpy as np

# python whole.py ./zoom4/sz1_70/test_latest/images/synthesized_image 512 png ./512r.png
# For vetical order:
# python whole.py /Users/light/Downloads/real 1024 png ./z1s_1024.png 1
imageSize = 256
# imageSize = 4096
# imageSize = 1024

def main(args):
    DIR = args[0]
    size = int(args[1])
    keyword = args[2]
    saveName = args[3]
    vertical = False
    if len(args) == 5:
        vertical = True
    if size % imageSize != 0:
        print('illegal size!')
        sys.exit()
    m = int(size / imageSize)
    images = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name)) and ".png" and keyword in name or '.jpg' in name]
    images.sort()
    imagesNumber = len(images)
    print(imagesNumber)
    # print(images)
    if imagesNumber % (m * m) != 0:
        print('illegal images number!')
        sys.exit()
    total = imagesNumber // (m * m)
    print(total)
    
    for h in range(total):
        if size == 4096:
            w = 0 + h * 256
            temp = np.zeros(shape=(4096, 4096, 3))
            temp1 = np.zeros(shape=(1024, 1024, 3))
            for i in range(0, 16):
                for k in range(0, 4):
                    for l in range(0, 4):
                        fileName = DIR + "/" + images[w]
                        # print(fileName)
                        img = cv2.imread(fileName)
                        temp1[k*imageSize:k*imageSize + imageSize, l*imageSize: l*imageSize+imageSize, :] = img
                        w += 1
                p = i//4
                q = i%4
                temp[p*1024: p*1024+1024, q*1024: q*1024+1024, :] = temp1
                # st = str(i) + ".png"
                # cv2.imwrite(st, temp)

            # cv2.imwrite("./whole/" + str(i) + ".png", temp)
            saveName1 = saveName[0:saveName.rfind('.')] + ".png"
            print(saveName1)
            cv2.imwrite(saveName1, temp)
            continue

        l = 0 + h * 16
        temp = np.zeros(shape=(size, size, 3), dtype=np.uint8)
        for j in range(0, m):
            for k in range(0, m):
                fileName = DIR + "/" + images[l]
                # print(fileName)
                img = cv2.imread(fileName)
                if vertical:
                    temp[(m-k-1) * imageSize: (m-k) * imageSize, j * imageSize: j * imageSize + imageSize, :] = img
                else:
                    temp[j * imageSize: j * imageSize + imageSize, k * imageSize: k * imageSize + imageSize, :] = img
                l += 1

        # cv2.imwrite("./whole/" + str(i) + ".png", temp)
        saveName1 = saveName[0:saveName.rfind('.')] + ".png"
        print(saveName1)
        cv2.imwrite(saveName1, temp)
            
    return 0

if __name__ == "__main__":
    # print(len(sys.argv))
    main(sys.argv[1:])