import cv2
import numpy as np

'''
Problem 1: Finding Hidden Objects
'''
# read the image
img_finding = cv2.imread('2.jpg') 
y_f, x_f = img_finding.shape[:2]

# crop input image
img_input = img_finding[0:y_f, 0:int(x_f*0.7), :]
img_gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
img_gray = cv2.bilateralFilter(img_gray, 10, 20, 20)

# crop template images
img_template = img_finding[int(y_f*0.07):y_f, int(x_f*0.7):x_f, :]
# convert the image to grayscale
img_template_gray = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)
# chuyển sang ảnh nhị phân với ngưỡng = 240
th, thres = cv2.threshold(img_template_gray, 240,255, cv2.THRESH_BINARY_INV)
# elliptical kernel with structuring element 11x11
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
morphed = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernel)
#find contours 
cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
#sử dụng cv.contourArea để tìm ra diện tích các contours và sắp xếp thứ tự diện tích từ cao xuống thấp
cnts_template = sorted(cnts, key=cv2.contourArea)[::-1]

k = 0
# finding use template matching
for i in cnts_template[:12]:
  k += 1
  x,y,z,t = cv2.boundingRect(i)
  cv2.putText(img_template, str(k), (x+z,y+t), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 3, color = (255,0,0), thickness = 3)
  template = img_template[y:y+t, x:x+z,:]
  template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
  template = cv2.bilateralFilter(template, 10, 20, 20)

  for scale in np.linspace(0.7, 1.0, 25)[::-1]:
    resized = cv2.resize(template, (int(template.shape[1] * scale), int(template.shape[0] * scale)))
    w, h = resized.shape[::-1]
    result = cv2.matchTemplate(img_gray, resized, cv2.TM_CCOEFF_NORMED)
    threshold = 0.52
    loc = np.where(result >= threshold)
    if loc:
      for pt in zip(*loc[::-1]):
        cv2.rectangle(img_finding, pt, (pt[0] + w, pt[1] + h), (0,255,0), 5)
        cv2.putText(img_finding, str(k), (pt[0] + w, pt[1] + h), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 3, color = (255,0,0), thickness = 3)
    if len(loc[0])!=0:
      break

# show result 
h,w = img_finding.shape[:2]
img_resize = cv2.resize(img_finding,(int(h/3), int(w/3)))
cv2.imshow('Finding Hidden Object', img_resize)
cv2.waitKey(0)


'''
Problem 2: Counting Rabbits
'''
# Opening image
img_counting = cv2.imread("rabbit.jpeg")

# grayscale and binary image
gray = cv2.cvtColor(img_counting, cv2.COLOR_BGR2GRAY)
_,thresh = cv2.threshold(gray, 185,255, cv2.THRESH_BINARY_INV)

# morphological operation
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
morphed_counting = cv2.morphologyEx(thresh, cv2.MORPH_RECT, element)
cnts_counting = cv2.findContours(morphed_counting.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

n = 0
for i in cnts_counting:
    if(cv2.contourArea(i)>4400):
        a,b,c,d = cv2.boundingRect(i)
        if (c<d or (c>d and cv2.contourArea(i)>7000)):
            cv2.rectangle(img_counting, (a,b), (a + c, b + d), (255, 0, 0), 5)
            cv2.putText(img_counting, str(n+1), (a -10 + int(c/2), b + int(d/2)), fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale=2, color=(0, 0, 0), thickness=2)
            n += 1

# show result
print('Have', n, 'rabbits')
d,c = img_counting.shape[:2]
img_counting = cv2.resize(img_counting,(int(d/1.5), int(c/1.5)))
cv2.imshow('Counting Rabbits', img_counting)
cv2.waitKey(0)