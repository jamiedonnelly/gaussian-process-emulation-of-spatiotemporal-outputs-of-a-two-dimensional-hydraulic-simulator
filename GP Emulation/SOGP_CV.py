import os 
import numpy as np 
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import GPy
import matplotlib.pyplot as plt
import time

def plot_pc(yts,preds,var,file):
    """
        Function for plotting principal component predictions
    """
    fig, ax = plt.subplots(2,3,figsize=(15,7))
    axs = ax.ravel()

    for i in range(len(axs)):
        axs[i].plot(yts[:,i],linewidth=3,color='black')
        axs[i].fill_between([i for i in range(len(yts))],(preds[:,i]-2*np.sqrt(var[:,i])),(preds[:,i]+2*np.sqrt(var[:,i])),alpha=0.25)
        axs[i].plot(preds[:,i],linestyle='dashed',linewidth=2)
        axs[i].legend(['True','Predicted','95% CI'])
    plt.savefig(file)

# Data indices for each of the hydrographs 

intervals =[   0,  124,  375,  485,  693,  937, 1061, 1312, 1422, 1630, 1874,
       1998, 2249, 2373, 2624]

# Load PCA model, outputs, inputs and data scaling parameters

with open("C:/.../pcmodel.pkl","rb") as f:
    pc = pickle.load(f)
with open("C:/.../ypc.npy","rb") as f:
    y=np.load(f)
with open("C:/.../X.npy","rb") as f:
    X=np.load(f)
with open("C:/.../scale_params.npy","rb") as f:
    scale = np.load(f)
    
# Loop to perform cross-validation and save the 1/k predictions to file 
    
for i in range(1,len(intervals)):
    # Train/Test datasets based on the hydrograph intervals 
    ts = list({i for i in range(intervals[i-1],intervals[i])})
    tr = list({i for i in range(len(y))}.difference(ts))
    xtr, xts = X[tr], X[ts]
    ytr, yts = y[tr], y[ts]
    
    preds = [] # store mean predictions
    var = [] # store predictive variances 
    
    # For each output train GP and generate predictions for i-th output feature
    for a in range(ytr.shape[1]):
        k = GPy.kern.Linear(xtr.shape[1])*GPy.kern.Matern32(xtr.shape[1]) # Matern 3/2 kernel function 
        m = GPy.models.GPRegression(xtr,ytr[:,a].reshape(len(ytr),1),k)
        m.optimize(messages=False)
        p = m.predict(xts)
        m2,s2 = p[0],p[1]
        preds.append(m2)
        var.append(s2)
    preds = np.concatenate([preds[j] for j in range(len(preds))],axis=1)
    var  = np.concatenate([var[j] for j in range(len(var))],axis=1)
    
    # Store images of predictions in latent feature (pca) space 
    if len(str(i))==1:
        file = os.path.join("C:/../Images/pc","preds0"+str(i))
    else:
        file = os.path.join("C:/../Images/pc","preds"+str(i))
    plot_pc(yts,preds,var,file)
    
    # Transform predictions from latent feature space back to original feature space
    itf = pc.inverse_transform(preds)
    
    # Write predictions to file 
    if len(str(i))==1:
        with open(os.path.join("C:/.../SOGPpreds","preds0"+str(i)),"wb") as f:
            np.save(f,itf)
    else:
        with open(os.path.join("C:/.../SOGPpreds","preds"+str(i)),"wb") as f:
            np.save(f,itf)
    del itf