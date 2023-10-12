#230730
import re
import glob
import os
import numpy as np
import scipy.io as scio
import skimage.io as skio
from utils_imageJ import save_tiff_imagej_compatible

def tif2np(filename):
    print(skio.imread(filename).shape, filename)
    return skio.imread(filename)

def calculate(filename,savename):
    img = tif2np(filename)[0]
    img = 1-img/65536
    img = (img-img.min())/(img.max()-img.min())
    img = img*0.011+0.985
    save_tiff_imagej_compatible(savename, img.astype(np.float32), "YX")


def div_save(filename1,filename2,save_name):
    imggt1 = skio.imread(filename1)
    imggt2 = skio.imread(filename2)
    res = imggt1/imggt2
def concat_on_off(dir1,dir2,save_name='/dataf/Research/Jax-AI-for-science/Data/matlab/kun_divide0911/all.tif'):
    os.makedirs(os.path.dirname(save_name),exist_ok=True)
    onlist= glob.glob(dir1)
    offlist= glob.glob(dir2)
    # imggt = np.zeros((25,2448,2048))
    imggt = []
    for onpath,offpath in zip(sorted(onlist),sorted(offlist)):
        imggt.append(skio.imread(onpath))
        # imggt.append(1-skio.imread(offpath))
    imggt = np.stack(imggt)
    save_tiff_imagej_compatible(save_name, imggt, "CYX")
  


def convert_tif_to_mat(data):
    tiflist= glob.glob(data)
    matlist = [os.path.dirname(x)+'_matlab/'+os.path.basename(x)[:-4]+'.mat' for x in tiflist]

    for (tifpath,matpath) in zip(tiflist,matlist):
        imggt = skio.imread(tifpath)
        wf = imggt.mean(0,keepdims=True)
        imggt = np.concatenate([wf,imggt],axis=0)
        os.makedirs(os.path.dirname(matpath),exist_ok=True)
        scio.savemat(matpath,{'imggt':imggt.astype(np.float32)})

        mat = scio.loadmat(matpath)
    #测试转换是否可逆
    img = mat['imggt']
    tifpath = matpath[:-4]+'.tif'
    img = img[1:]
    print((img-imggt[1:]).sum())
    # os.makedirs(os.path.dirname(tifpath), exist_ok=True)
    # skio.imsave(tifpath, img.astype(np.uint16))
def stackandconvert(fileStr,new_name):
    tiflist= glob.glob(fileStr)
    imggt = np.zeros((25,2448,2048))
    for i,tifpath in enumerate(sorted(tiflist)):
        imggt[i] = skio.imread(tifpath)
    wf = imggt.mean(0,keepdims=True)
    imggt = np.concatenate([wf,imggt],axis=0)
    os.makedirs(os.path.dirname(new_name),exist_ok=True)
    scio.savemat(new_name,{'imggt':imggt.astype(np.float32)})
if __name__ == '__main__':
    # stackandconvert('/home/wtxt/share/on&off/on/*.tif',"Data/matlab/kun_on&off/on.mat")
    # stackandconvert('/home/wtxt/share/on&off/off/*.tif',"Data/matlab/kun_on&off/off.mat")
    convert_tif_to_mat('/home/wtxt/a/data/kun_original0911/*.tif')
    # concat_on_off('/home/wtxt/share/xx/*.tif','/home/wtxt/share/off/*.tif',)
    # calculate('/home/wtxt/a/data/kun_original0911/','/home/wtxt/a/data/kun_original0911/')