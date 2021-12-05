import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from sklearn.decomposition import PCA 
import pickle
import os
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time

shape = (854,1026)

# Load test directories and prediction directories 
testdir = "C:/.../cvtest"
tests = [os.path.join(testdir,i) for i in os.listdir(testdir)]
preddir = "C:/.../preds"
preds = [os.path.join(preddir,i) for i in os.listdir(preddir) if "Images" not in i]

# Collect results: RMSE, f1
rmse = []
maxtrue = []
maxpred = []
f1 = []
for i in range(len(tests)):
    with open(tests[i],"rb") as f:
        ytest = np.load(f)
    maxtrue.append(np.max(ytest,axis=0).reshape(shape))
    with open(preds[i],"rb") as f:
        pred = np.load(f)
    pred[pred<0.1]=0
    maxpred.append(np.max(pred,axis=0).reshape(shape))
    rmse.append(np.sqrt(mean_squared_error(ytest,pred)))
    f1y=ytest.flatten()
    f1p=pred.flatten()
    f1y[f1y<=0.3]=0
    f1y[f1y>0.3]=1
    f1p[f1p<=0.3]=0
    f1p[f1p>0.3]=1
    f1.append(f1_score(f1y,f1p))
print("Mean RMSE:",np.mean(rmse))
print("Median RMSE:",np.median(rmse))

# Plot results 
fig, ax = plt.subplots(1,2,figsize=(16,7))
ax[0].scatter([i for i in range(1,len(rmse)+1)],rmse,color='black')
ax[0].hlines(np.mean(rmse),xmin=1,xmax=14,linestyle='dashed',linewidth=1,color='red')
ax[0].hlines(np.median(rmse),xmin=1,xmax=14,linestyle='dashed',linewidth=1)
ax[0].set_xlabel("Hydrograph")
ax[0].set_xticks([i for i in range(1,len(rmse)+1)])
ax[0].set_ylabel("RMSE")
ax[0].set_title("RMSE LOOCV scores")
ax[0].legend(['Scores','Mean RMSE','Median RMSE'])

ax[1].scatter([i for i in range(1,len(f1)+1)],f1,color='black')
ax[1].hlines(np.mean(f1),xmin=1,xmax=14,linestyle='dashed',linewidth=1,color='red')
ax[1].hlines(np.median(f1),xmin=1,xmax=14,linestyle='dashed',linewidth=1)
ax[1].set_xlabel("Hydrograph")
ax[1].set_xticks([i for i in range(1,len(f1)+1)])
ax[1].set_ylabel("F1")
ax[1].set_title("F1 LOOCV scores")
ax[1].legend(['Scores','Mean F1','Median F1'])
plt.show()
