import numpy
import cv2
import face_recognition

imgnan = face_recognition.load_image_file('images/nandeesh.jpg')
imgnan = cv2.cvtColor(imgnan, cv2.COLOR_BGR2RGB)
imgtest = face_recognition.load_image_file('images/nandeesh test.jpg')
imgtest = cv2.cvtColor(imgtest, cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgnan)[0]
encodenan = face_recognition.face_encodings(imgnan)[0]
cv2.rectangle(imgnan, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (255, 0, 255), 2)

faceloctest = face_recognition.face_locations(imgtest)[0]
encodetest = face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest, (faceloctest[3], faceloctest[0]), (faceloctest[1], faceloctest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodenan], encodetest)
facedis = face_recognition.face_distance([encodenan], encodetest)
print(results, facedis)
cv2.putText(imgtest, f'{results} {round(facedis[0],2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (1, 1, 255), 2)

cv2.imshow('Nandeesh', imgnan)
cv2.imshow('test', imgtest)
cv2.waitKey(0)
