import numpy as np
import pandas as pd
from PIL import Image
#Read data from csv file
train_file = "train.csv"
data = pd.read_csv(train_file)
images = data.iloc[:,1:].values
images = images.astype(np.float)
labels = data[[0]].values.ravel()
image_size = images.shape[1]
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
for x in range(0,34000, 2000):
    nTrain = np.empty((1,785))
    for i in range(x,x+2000):
       lab = labels[i]
       for r in range(-10,20,10):
            image = Image.fromarray(images[i].reshape(28,28))
            #print(image)
            rotated = Image.Image.rotate(image, r)
            temp = np.roll(np.array(rotated), -3, axis=1)
            a = np.array(temp.flatten())
            b = (np.insert(a,0,lab)).reshape(1,785)
            nTrain = np.append(nTrain,b,axis=0)
       print(i)
    np.savetxt("trans_new%s.csv" % x, nTrain, delimiter=",");
