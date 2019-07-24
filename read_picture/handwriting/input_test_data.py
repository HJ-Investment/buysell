import cv2
import os

path = '/tf/handwriting/datasets/testdata/'
files = os.listdir(path)

for file in files:
    if file.startswith('.'):
        continue
    if not os.path.isdir(path + file):
        print(file)
        image = cv2.imread(path + file)

        height, width, channels = image.shape

        print(height, width, channels)

        if height > width:
            delta = height - width
            img = cv2.copyMakeBorder(image, 0, 0, 0, delta,cv2.BORDER_CONSTANT,value=[0,0,0])
            img = cv2.resize(img, (224,224))
            cv2.imwrite('./handwriting/datasets/testdata/fixed/'+file.replace('.png','_')+'fixed.png', img)
        else:
            delta = width - height
            img = cv2.copyMakeBorder(image, 0, delta, 0, 0,cv2.BORDER_CONSTANT,value=[0,0,0])
            img = cv2.resize(img, (224,224))
            cv2.imwrite('./handwriting/datasets/testdata/fixed/'+file.replace('.png','_')+'fixed.png', img)


