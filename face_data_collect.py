import cv2
import numpy as np

#INIT Camera
cap = cv2.VideoCapture(0)#0 is id

#face detection
face_cascade = cv2.CascadeClassifier("C:\\Users\\Tarun\\AppData\\Roaming\\Python\\Python37\\site-packages\\cv2\\data\\haarcascade_frontalface_alt.xml")



skip=0
face_data = []
dataset_path = 'E:\\Data\\'


file_name = input("Enter the name of the person : ")
while True:
    ret,frame = cap.read()

    if ret==False:
        continue


    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)



    faces = face_cascade.detectMultiScale(frame,1.3,5)
    faces = sorted(faces,key=lambda f:f[2]*f[3])

    #pick largest face acc to area(last face)
    for face in faces[-1:]:
        x,y,w,h=face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

        #extract region of interest--crop out required face
        offset = 10#padding
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))

        #Store every 10th Face
        if skip%10==0:
            face_data.append(face_section)
            print(len(face_data))
    

    cv2.imshow("Frame",frame)
    cv2.imshow("Face_frame",face_section)
                  
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

#convert our list to a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

#save this data into file system
np.save(dataset_path+file_name+'.npy',face_data)
print("Data Suceesfully saved at "+dataset_path+file_name)

cap.release()
cv2.destroyAllWindows()
