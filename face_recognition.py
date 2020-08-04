import cv2
import numpy as np
import os




######### KNN  #########
def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))


def knn(X,Y,k=5):

    vals = []
    m = X.shape[0]

    for i in range(m):
        ix = X[i,:-1]
        iy = X[i,-1]
        d = dist(Y,ix)
        vals.append((d,iy))

    vals = sorted(vals,key=lambda x:x[0])
    #nearest/first k points
    vals = vals[:k]

    vals = np.array(vals)

    #print(vals)
    new_vals = np.unique(vals[:,1],return_counts=True)
    print(new_vals)
    index = np.argmax(new_vals[1])
    pred = new_vals[0][index]

    return pred
##############################






#INIT Camera
cap = cv2.VideoCapture(0)#0 is id

#face detection
face_cascade = cv2.CascadeClassifier("C:\\Users\\Tarun\\AppData\\Roaming\\Python\\Python37\\site-packages\\cv2\\data\\haarcascade_frontalface_alt.xml")

skip=0
face_data = []
labels = []
dataset_path = 'E:\\Data\\'

class_id = 0
names={} #mapping btw id and name


#data preparation
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        #create mapping bw class_id and name
        names[class_id] = fx[:-4]
        print("loaded"+fx)
        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)

        #create labels for the class
        target = class_id*np.ones(data_item.shape[0],)
        class_id+=1
        labels.append(target)

face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(labels,axis=0).reshape((-1,1))
print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)


#testing

while True:
    ret,frame = cap.read()

    if ret==False:
        continue


    faces = face_cascade.detectMultiScale(frame,1.3,5)
    faces = sorted(faces,key=lambda f:f[2]*f[3])

    for face in faces:
        x,y,w,h=face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

        #extract region of interest--crop out required face
        offset = 10#padding
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))

        #predict label(out)
        out = knn(trainset,face_section.flatten())

        #display around the screen
        pred_name = names[int(out)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)


    cv2.imshow("Faces",frame)
                  
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
