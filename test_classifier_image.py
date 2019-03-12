import keras
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

persons = os.listdir('datasets/')[1:]
print(persons)
font = cv2.FONT_HERSHEY_SIMPLEX

model = keras.models.load_model('CNN_MODEL_LAST')

file_name = '1.jpg'
img = cv2.imread(file_name)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3,5)
print("faces found : {}".format(len(faces)))
count = 0

print("Press ESC to terminate....")
for (x,y,w,h) in faces:
	count += 1
	face = img[y:y+h, x:x+w]
	draw_rec = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
	gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
	resize_face = cv2.resize(gray_face,(50,50)).reshape(-1,50,50,1)

	test_predict = model.predict(resize_face, batch_size = 64, verbose = 0)
	

	#predicting face
	predict_face = model.predict_classes(resize_face, verbose = 0)

	label = persons[predict_face[0]]
	for i in test_predict:
		print(i)
		conf = i[predict_face[0]]
		if conf < 0.70:
			label = "Unknown"
	cv2.putText(img,label.title(), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (51,255,255), lineType=cv2.LINE_AA) 

	#putting the label on the picture
	#cv2.putText(img,label.title(), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (51,255,255), lineType=cv2.LINE_AA) 


cv2.imshow('image', draw_rec)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Program terminated")







