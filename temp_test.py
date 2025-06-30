import rasterio
from datetime import datetime
time_data = '20220901'
test = datetime.strptime(time_data, '%Y%m%d')
print(test)


"""
file = '/media/DATA/ADeWit/3STR/dataset/SEN12MSCRTS/ROIs1970/21/S2/0/s2_ROIs1970_21_ImgNo_0_2018-01-06_patch_250.tif'
with rasterio.open(file) as src:
    data = src.read() 
    print(data.shape)
    print(data.dtype)
    print(data.max(), data.min())
    print(data[0, 0, 0:10])
    print(data[5, 0, -10:])
    print(data[12, 0, 0:10])

"""