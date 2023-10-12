import numpy as np
import rasterio
from PIL import Image
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from PIL import Image


# 将numpy array转换为tif文件并保存
# img_resized = Image.fromarray((img_resized * 255).astype(np.uint8))
# img_resized.save('output.tif')
tif_path1 = '/home/wtxt/a/data/kun_original0911_matlab/Aligned off blur_original.tif'
tif_path2 = '/home/wtxt/a/data/kun_original0911_matlab/Aligned on blur.tif'

# 读取tif文件为numpy array
img = imread(tif_path1)[0]
print(img.shape)
# 将图像长宽缩放为原来的二倍
img_resized = resize(img, (img.shape[0]*2, img.shape[1]*2))
# 1. 读取tif文件
with rasterio.open(tif_path2) as src:
    band = src.read(1)
print(band.shape)
# 2. 将数据归一化到0-1之间
def torgb(band):
    band_normalized = (band - band.min()) / (band.max() - band.min())

    # 3. 归一化后的数据乘以255，然后将其转换为uint8
    band_normalized = (255*band_normalized).astype(np.uint8)
    return band_normalized
    # 4. 创建RGB图像，因为原图像只有一通道，所以我们将同一通道赋值给R, G, B三通道
    return np.dstack((band_normalized, band_normalized, band_normalized))

# 5. 保存为tif文件
im1,im2 = torgb(img_resized),torgb(band)
print(im1.shape, im2.shape)
from utils_imageJ import save_tiff_imagej_compatible
im = np.stack([im1,im2])
save_name = '/home/wtxt/a/data/kun_original0911_matlab/output1.tif'
save_tiff_imagej_compatible(save_name, im, "CYX")