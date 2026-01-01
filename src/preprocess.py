import cv2
import os
import imutils

IMG_SIZE = (224, 224)

def crop_brain(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, 2)
    thresh = cv2.dilate(thresh, None, 2)

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if cnts:
        c = max(cnts, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        img = img[y:y+h, x:x+w]

    return cv2.resize(img, IMG_SIZE)

def preprocess_dataset(input_dir):
    for cls in os.listdir(input_dir):
        cls_path = os.path.join(input_dir, cls)
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = crop_brain(img)
            cv2.imwrite(img_path, img)

if __name__ == "__main__":
    preprocess_dataset("data/raw")
