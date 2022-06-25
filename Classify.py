import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import PIL.ImageOps
from PIL import Image

x,y = fetch_openml('mnist_784',version = 1,return_X_y = True)
x = np.array(x)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state = 0, train_size = 7500, test_size = 2500)
xtrain = xtrain/255.0
xtest = xtest/255.0

model = LogisticRegression(solver = 'saga', multi_class = "multinomial").fit(xtrain, ytrain)

def getPred(img):        
    im_pil = Image.open(img)
    img_bw = im_pil.convert('L')
    img_resized = img_bw.resize((28,28), Image.ANTIALIAS)
    min_pixel = np.percentile(img_resized, 20)
    img_scaled  = np.clip(img_resized - min_pixel, 0, 255)
    max_pixel = np.max(img_resized)
    img_scaled = np.asarray(img_scaled)/max_pixel
    test_sample = np.array(img_scaled).reshape(1, 784)
    test_pred = model.predict(test_sample)
    return test_pred[0]