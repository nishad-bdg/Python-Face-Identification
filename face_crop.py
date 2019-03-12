import os
import numpy as np
import cv2


DATA_DIR  = 'data/'
persons = os.listdir(DATA_DIR)[1:]
print(persons)
SAVE_PATH = 'datasets/'


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

for person in persons:
	path = os.path.join(DATA_DIR,person)
	class_num = persons.index(person)

	for img in os.listdir(path):
		try:
			img_array = cv2.imread(os.path.join(path,img))
			gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
			faces = face_cascade.detectMultiScale(gray, 1.3,5)
			for(x,y,w,h) in faces:
				face_crop = img_array[y:y+h, x:x+w ]
				#face_img = cv2.rectangle(img_array,(x,y),(x+w,y+h),(0,255,0),2)
				save_path = 'datasets/{}/{}'.format(person,img)
				cv2.imwrite(save_path,face_crop)
		except:
			print(img)
			print("Unable to convert image")
		







