import numpy as np
import cv2
import keras
import os

model = keras.models.load_model('CNN_MODEL_LAST')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX
persons1 = os.listdir('datasets/')
#print(persons1)
persons = os.listdir('datasets/')[1:]

print("Press ESC to terminate....")

is_cap = True

while is_cap:
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3,5)

	for(x,y,w,h) in faces:
		#print("Face detected : {}".format(len(faces)))
		face_crop = img[y:y+h, x:x+w]
		#print(face_crop)
		cv2.rectangle(img, (x,y),(x+w, y+h), (0,255,0),2)
		gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
		resize_face = cv2.resize(gray_face,(50,50)).reshape(-1,50,50,1)

		test_predict = model.predict(resize_face,batch_size = 64, verbose = 0)

		predict_face = model.predict_classes(resize_face, batch_size = 64, verbose = 0)
		


		
		label = persons[predict_face[0]] 

		for i in test_predict:
			print(i)
			conf = i[predict_face[0]]
			print(conf)
			if conf < 1:
				label = "Unknown"

		cv2.putText(img, label, (x,y), font, 0.8,(0,255,0),2,cv2.LINE_AA)
	cv2.imshow('Image',img)
	
	k = cv2.waitKey(1) & 0xFF

	# if user presses ESC key program will terminate
	if k == 27:
		break
cap.release()
cv2.destroyAllWindows()
print("Program terminated....")
