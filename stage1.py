import numpy as np
from osgeo import gdal, gdal_array
import pandas as pd
import operator
import gc
import os

roi_ds = gdal.Open('dataset.tif', gdal.GA_ReadOnly)

roi = roi_ds.GetRasterBand(1).ReadAsArray()

# How many pixels are in each class?
classes = np.unique(roi)

dict = {}

for c in classes:
    dict[c] = (roi == c).sum()
sorted_x = sorted(dict.items(), key=operator.itemgetter(1))
print("Top 6 classes and pixel counts \n",sorted_x[-6:])

# Find how many non-zero entries we have -- i.e. how many training data samples?
n_samples = (roi > 0).sum()
print('We have {n} samples'.format(n=n_samples))

# What are our classification labels?
labels = np.unique(roi[roi > 0])
print('The training data include {n} classes: {classes}'.format(n=labels.size,classes=labels))
# We will need a "X" matrix containing our features, and a "y" array containing our labels
#     These will have n_samples rows
#     In other languages we would need to allocate these and them loop to fill them, but NumPy can be faster

#X = img_b1[roi > 0, :]  
y = roi[roi > 0]
print("Printing y = ",y)

images = ['satellite_img.tif']
final = pd.DataFrame()


for c in classes:
    temp = pd.DataFrame()  # Create a temporary DataFrame for each class
    
    
    for img in images:
        #print(img)
        
        train_ds = gdal.Open(img, gdal.GA_ReadOnly)
        #print(train_ds.RasterXSize, train_ds.RasterYSize)
        
        img_b1 = np.zeros((train_ds.RasterYSize, train_ds.RasterXSize, train_ds.RasterCount),
                          gdal_array.GDALTypeCodeToNumericTypeCode(train_ds.GetRasterBand(1).DataType))
        
        for b in range(img_b1.shape[2]):
            img_b1[:, :, b] = train_ds.GetRasterBand(b + 1).ReadAsArray()
        
        #print(img_b1.shape)

        # Assuming `roi` is a binary mask with the same shape as img_b1[:, :, 0]
        roi = np.random.choice([0, 1], size=(img_b1.shape[0], img_b1.shape[1]))

        Xt = img_b1[roi == c, :]
        Xt1 = pd.DataFrame(Xt)
        
        # Limit the number of samples to 1,00,000
        Xt2 = Xt1.sample(n=min(100000, Xt1.shape[0]))
        
        Xt2.reset_index(drop=True, inplace=True)
        
        temp = pd.concat([Xt2, temp], axis=1)
        temp["class"] = c
    
    final = pd.concat([temp, final], axis=0)
    final.reset_index(drop=True, inplace=True)
    
    
    gc.collect()

#print the shape of the final dataset
print("Final Shape = ",final.shape)


final.columns = ['col_'+str(i) for i in range(final.shape[1])]
#print the head of the final dataset
print(final.head())


#print final.head to csv
final.to_csv('final.csv', index=False)


#----Train and test split----

# encode class values as integers


