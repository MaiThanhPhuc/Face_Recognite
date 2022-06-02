# -----------------------------------------------------------------------------------------
# Code base on http://krasserm.github.io/2018/02/07/deep-face-recognition/
# -----------------------------------------------------------------------------------------

#%% import thư viện với một số module
from asyncio.windows_events import NULL
from model import create_model
import cv2
import numpy as np
import os
from align import AlignDlib
import joblib

from sklearn.metrics import accuracy_score, euclidean_distances

#%% Define func
class IdentityMetadata():
    def __init__(self, base, name, file):
        # Thư mục chứa dữ liệu (image)
        self.base = base #image
        # Thư mục con của từng mẫu (NameOfSample)
        self.name = name #19110015 -> label
        # Các sample của mẫu
        self.file = file #19110015_0001.bmp -> train data
    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file) 


def load_image(path):
    img = cv2.imread(path, 1)
    # in BGR order. So we need to reverse them
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def load_metadata(path):
    metadata = []
    for i in sorted(os.listdir(path)):
        for f in sorted(os.listdir(os.path.join(path, i))):
            # Cho phép jpg, jpeg, bmp
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == '.jpeg' or ext == '.bmp':
                metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)

load_metadata("image")

def load_metadata_one(path,name):
    metadata = []
    for i in sorted(os.listdir(os.path.join(path, name))):
        # Cho phép jpg, jpeg, bmp
        ext = os.path.splitext(i)[1]
        if ext == '.jpg' or ext == '.jpeg' or ext == '.bmp':
            metadata.append(IdentityMetadata(path, name, i))
    return np.array(metadata)

def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img), 
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

alignment = AlignDlib('models/shape_predictor_68_face_landmarks.dat')
#cái này là để tìm 68 điểm làm landmark trên khuôn mặt, từ đó detect ra khuôn mặt


nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')


#%%
#Tạo thư mục chứa data
parent_dir = "image"
pathName = input("Nhap pathname: ")
path = os.path.join(parent_dir, pathName)
os.mkdir(path)
print("Directory '%s' created" %pathName)



#%%
#------------------- Get data ---------------------
#640x480
# define a video capture object
vid = cv2.VideoCapture(0)
count = 1
n = 0
while(count<=100):
    [success, frame]  = vid.read()
    cv2.rectangle(frame, (195, 50), (251+195, 251+115), (0,255,0),1)
    if success:
            imgROI = frame[115:250+115,195:(195+250)] #tạo ra ảnh
            tempImg = align_image(imgROI)
            if tempImg is None:
                frame = cv2.putText(frame,"Can't detect face!", (185,45), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 2)
                frame = cv2.putText(frame,"Put face into rectangle", (125,250+140), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 2)
            else:
                frame = cv2.putText(frame,"Ok, hold", (185,45), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2)
                if n == 5:
                    filename = 'image/%s/%s_%04d.bmp'%(pathName,pathName,count)
                    print(filename)
                    cv2.imwrite(filename,imgROI)
                    count+=1
                    n=0
                else: n+=1
            cv2.imshow('frame', frame)       
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Destroy all the windows
vid.release()
cv2.destroyAllWindows()

#%% Load data, xử lý và training, ở đây ta sẽ dùng model đã được train sẵn nn4_small2 để representation hay embedded ảnh => 128D  
metadata = load_metadata('image')
embedded = np.zeros((metadata.shape[0], 128))

for i, m in enumerate(metadata):
    img = load_image(m.image_path())
    print(m.image_path())
    img = align_image(img)
    # features scaling 0 -> 1
    img = (img / 255)
    # obtain embedding vector for image
    # phải expand thêm 1 chiều vì predict của model pre-trained yêu cầu
    #chuyển, represent, embedded 
    embedded[i] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
import pickle as pkl
filename = 'embeded'
fileObject = open(filename, 'wb')
pkl.dump(embedded, fileObject)
# fileObject2 = open(fileName, 'wb')
# embedded = pkl.load(fileObject2)
fileObject.close()
#%%

y = np.array([m.name for m in metadata])
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(y)
labels = encoder.transform(y)
#1/2 : 1/2
train_idx = np.arange(metadata.shape[0]) % 2 != 0
test_idx = np.arange(metadata.shape[0]) % 2 == 0

X_train = embedded[train_idx]
X_test = embedded[test_idx]

y_train = labels[train_idx]
y_test = labels[test_idx]

#Import một số model classification
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier       
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

lSvc_clf = LinearSVC()
svc_clf = SVC()
sgd_clf = SGDClassifier(random_state=42) 
ova_clf = OneVsRestClassifier(SGDClassifier(random_state=42))
knn_clf = KNeighborsClassifier(metric='euclidean')
forest_clf = RandomForestClassifier(n_estimators=1000, random_state=42)

lSvc_clf.fit(X_train, y_train)
svc_clf.fit(X_train, y_train)
sgd_clf.fit(X_train, y_train)
ova_clf.fit(X_train, y_train)
knn_clf.fit(X_train, y_train)
forest_clf.fit(X_train, y_train)

acc_lSvc_clf = accuracy_score(y_test, lSvc_clf.predict(X_test))
acc_svc = accuracy_score(y_test, svc_clf.predict(X_test))
acc_sgd = accuracy_score(y_test, sgd_clf.predict(X_test))
acc_ova = accuracy_score(y_test, ova_clf.predict(X_test))
acc_knn = accuracy_score(y_test, knn_clf.predict(X_test))
acc_forest = accuracy_score(y_test, forest_clf.predict(X_test))



print(f'Linear SVC accuracy = {acc_lSvc_clf}')
print(f'SVC accuracy = {acc_svc}')
print(f'SGD accuracy = {acc_sgd}')
print(f'OVA accuracy = {acc_ova}')
print(f'KNN accuracy = {acc_knn}')
print(f'Forest accuracy = {acc_forest}')

svc = SVC()
from sklearn.model_selection import cross_val_score
c_A = cross_val_score(svc, embedded, y, cv=5)
#%% Save model
joblib.dump(lSvc_clf,'lSvc_clf.pkl')
joblib.dump(svc_clf,'svc_clf.pkl')
joblib.dump(sgd_clf,'sgd_clf.pkl')
joblib.dump(ova_clf,'ova_clf.pkl')
joblib.dump(knn_clf,'knn_clf.pkl')
joblib.dump(forest_clf,'forest_clf.pkl')

#%% Tuning KNN
from sklearn.model_selection import GridSearchCV

param_grid = {'weights': ["uniform", "distance"], 'n_neighbors':np.append(np.arange(1,10,1),[10,15,20])}#np.arange(0,100,1)
knn_new = KNeighborsClassifier()

grid_search = GridSearchCV(knn_new, param_grid, cv=10, verbose=3)
grid_search.fit(X_train, y_train)
grid_search.best_params_    
grid_search.best_score_
joblib.dump(grid_search,'knn_clf.pkl')

#%% Tuning Random forest
param_grid = {
 'n_estimators': [100, 200, 400, 1000, 1200, 1600, 2000]}
forest_new = RandomForestClassifier()
grid_search = GridSearchCV(forest_new, param_grid, cv=3, verbose=3)
grid_search.fit(X_train, y_train)
grid_search.best_params_    
grid_search.best_score_

#%% Tuning SVC 
param_grid = {'C':np.append(np.arange(0,20,1),[0.1,0.5,100,200,500,1000]),'gamma': ['scale','auto'],'kernel': ['rbf', 'poly', 'sigmoid']}
#đối với default kernel sẽ mặc định là rbf
#{'gamma': ['scale','auto'] ,'C':np.arange(0.0, 4.0, 0.1)}
svc_clf = SVC() #default 

grid_search = GridSearchCV(svc_clf, param_grid, cv=5, verbose=3)
grid_search.fit(X_train, y_train)
grid_search.best_params_    
grid_search.best_score_

#%% Tuning SVC Linear 
param_grid = {'C':np.append(np.arange(0,20,1),[0.1,100,200,500,1000])}
#đối với default kernel sẽ mặc định là rbf
lSvc_clf = LinearSVC() #default 

grid_search = GridSearchCV(lSvc_clf, param_grid, cv=5, verbose=3)
grid_search.fit(X_train, y_train)
grid_search.best_params_    
grid_search.best_score_
#%% One largess face
svc_clf = joblib.load('svc_clf.pkl')
vid = cv2.VideoCapture(0)
while(1):
    [success, frame]  = vid.read()
    cv2.rectangle(frame, (194, 50), (251+195, 251+115), (0,255,0),1)
    #cv2.rectangle(frame, (194, 114), (251+195, 251+115), (0,255,0),1)
    
    if success:
            imgROI = frame[51:250+115,195:(195+250)] #tạo ra ảnh
            #imgROI = frame[115:250+115,195:(195+250)] #tạo ra ảnh
            tempImg = align_image(imgROI)
            if tempImg is None:
                frame = cv2.putText(frame,"Can't detect face!", (185,45), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 2)
                frame = cv2.putText(frame,"Put face into rectangle", (125,250+140), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 2)
            else:
                # scale RGB values to interval [0,1]
                tempImg = (tempImg / 255)
                # obtain embedding vector for image
                embedded_test = nn4_small2_pretrained.predict(np.expand_dims(tempImg, axis=0))[0] #Phải 
                pre = svc_clf.predict([embedded_test])[0]
                #arr = svc_clf.decision_function([embedded_test])[0]
                
                stringFrame = '%s'%(pre)
                frame = cv2.putText(frame,stringFrame, (185,45), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0,255,0), 2)
            cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Destroy all the windows
vid.release()
cv2.destroyAllWindows()

#multi face video

#%% Multi face
'''
count = 0
vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)
while(1):
    [success, frame]  = vid.read()
    
    if success:
            imgROI = frame
            tempImg = align_image(imgROI)
            if tempImg is None:
                frame = cv2.putText(frame,"Can't detect face!", (185,45), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 2)
            else:
                dets = alignment.detector(frame, 0)
                print(dets)
                for rect in dets:
                    print(dets,"inside")
                    x = rect.left()
                    y = rect.top()
                    w = rect.right()
                    h = rect.bottom()
                    print(x,y,w,h)
                    #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
                    cv2.rectangle(frame, (x-1,y-1),(w,h+1), (0, 255, 0), 1)	
                    imgROI = frame[y:h+1,x:w]
                    imgROI = cv2.resize(imgROI,(96,96))
                    tempImg = (tempImg / 255)
                    # obtain embedding vector for image
                    embedded_test = nn4_small2_pretrained.predict(np.expand_dims(imgROI, axis=0))[0]
                    pre = svc_clf.predict([embedded_test])[0]
                    arr = svc_clf.decision_function([embedded_test])[0]
                    dec = abs(arr.max())
                    if dec < 0.37:   
                        dec = (1 - dec/0.37)*100
                        stringFrame = '%s %.2f%%'%(pre,round(dec,2))
                        frame = cv2.putText(frame,stringFrame, (x,y), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2)
                    else:
                        frame = cv2.putText(frame,"Can't classification", (x,y), cv2.FONT_HERSHEY_DUPLEX, .5, (0,255,0), 2)
            cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Destroy all the windows
vid.release()
cv2.destroyAllWindows()
'''
####
#%%
import matplotlib.pyplot as plt
def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))
from sklearn.metrics import f1_score, accuracy_score
from tkinter import Frame, Tk, BOTH, Text, Menu, END
from tkinter.filedialog import Open, SaveAs
svc_clf = joblib.load('svc_clf.pkl')
my_result = np.array(sorted(os.listdir("image")))
class Main(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.initUI()
  
    def initUI(self):
        self.parent.title("Nhan Dang Khuon Mat")
        self.pack(fill=BOTH, expand=1)
  
        menubar = Menu(self.parent)
        self.parent.config(menu=menubar)
  
        fileMenu = Menu(menubar)
        fileMenu.add_command(label="Open", command=self.onOpen)
        fileMenu.add_command(label="Recognition", command=self.onRecognition)
        fileMenu.add_separator()
        fileMenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=fileMenu)
        self.txt = Text(self)
        self.txt.pack(fill=BOTH, expand=1)
  
    def onOpen(self):
        global ftypes
        ftypes = [('Images', '*.jpg *.tif *.bmp *.gif *.png')]
        dlg = Open(self, filetypes = ftypes)
        fl = dlg.show()
  
        if fl != '':
            global img
            global imgin
            imgin = cv2.imread(fl,cv2.IMREAD_COLOR)
            img = imgin[...,::-1]
            cv2.namedWindow("ImageIn", cv2.WINDOW_AUTOSIZE)
            #cv2.moveWindow("ImageIn", 200, 200)
            cv2.imshow("ImageIn", imgin)

    def onRecognition(self):
        print("Recognition")
        if align_image(imgin) is None:
            cv2.putText(imgin,"Non face in picture", (5,15), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 2)
            print("Non face in picture")
        # scale RGB values to interval [0,1]
        else:
            dets = alignment.detector(imgin, 0)
            print("inside",dets)
            for rect in dets:
                x = rect.left()
                y = rect.top()
                w = rect.right()
                h = rect.bottom()
                cv2.rectangle(imgin, (x-1,y-1),(w,h+1), (0, 255, 0), 1)
                imgROI = img[y:h+1,x:w]
                imgROI = cv2.resize(imgROI,(96,96))
                img_face = (imgROI / 255)
            # obtain embedding vector for image
                embedded_test = nn4_small2_pretrained.predict(np.expand_dims(img_face, axis=0))[0]
                pre = my_result[svc_clf.predict([embedded_test])[0]]
                print(pre)
                preMetaData = load_metadata_one("image",pre)
                minDistance = np.inf
                num = len(preMetaData)
                preEmbedded = np.zeros((preMetaData.shape[0], 128))
                for i, m in enumerate(preMetaData):
                    tImg = load_image(m.image_path())
                    print(m.image_path())
                    tImg = align_image(tImg)
                    tImg = (tImg / 255)
                    preEmbedded[i] = nn4_small2_pretrained.predict(np.expand_dims(tImg, axis=0))[0]
                for i in range(num):
                    tempDistance = distance(embedded_test,preEmbedded[i])
                    print(tempDistance)
                    if(tempDistance < minDistance):
                        minDistance = tempDistance
                    if(minDistance<0.52): break
                print(minDistance)
                if(minDistance < 0.52):
                    stringFrame = '%s '%(pre)
                    cv2.putText(imgin,stringFrame, (x,y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0), 2)
                else: 
                    cv2.putText(imgin,"Can't classified", (x,y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0), 2)
                #arr = svc_clf.decision_function([embedded_test])[0]
                #dec = abs(arr.max())
                #print(pre,arr,dec)
                
            cv2.namedWindow("ImageIn", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("ImageIn", imgin)
root = Tk()
Main(root)
root.geometry("450x450+100+100")
root.mainloop()

###
#%% Tính best score
import matplotlib.pyplot as plt
def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))
from sklearn.metrics import f1_score, accuracy_score

distances = [] # squared L2 distance between pairs
identical = [] # 1 if same identity, 0 otherwise

num = len(metadata)

for i in range(num - 1):
    for j in range(i + 1, num):
        distances.append(distance(embedded[i], embedded[j]))
        identical.append(1 if metadata[i].name == metadata[j].name else 0)
        
distances = np.array(distances)
identical = np.array(identical)

thresholds = np.arange(0.3, 1.0, 0.01)

f1_scores = [f1_score(identical, distances < t) for t in thresholds]
acc_scores = [accuracy_score(identical, distances < t) for t in thresholds]

opt_idx = np.argmax(f1_scores)
# Threshold at maximal F1 score
opt_tau = thresholds[opt_idx]
# Accuracy at maximal F1 score
opt_acc = accuracy_score(identical, distances < opt_tau)

# Plot F1 score and accuracy as function of distance threshold
plt.plot(thresholds, f1_scores, label='F1 score')
plt.plot(thresholds, acc_scores, label='Accuracy')
plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
plt.title(f'Accuracy at threshold {opt_tau:.2f} = {opt_acc:.3f}')
plt.xlabel('Distance threshold')
plt.legend()

# %% Đưa lên 2D để xem thoi
import matplotlib.pyplot as plt
targets = np.array([m.name for m in metadata])

from sklearn.manifold import TSNE

X_embedded = TSNE(n_components=2).fit_transform(embedded)

for i, t in enumerate(set(targets)):
    idx = targets == t
    plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)   

plt.legend(bbox_to_anchor=(1, 1))



# %%
