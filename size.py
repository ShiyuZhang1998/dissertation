#!/usr/bin/env python
# coding=utf-8
'''
Author: Lawrence1999
Date: 2021-09-06 17:23:55
Last Update: 2021-09-12 11:41:13
'''

# 1 用体素表示数据;
# 2 用点云表示 ;(体素大小需要测试)
# 3 上色依据点云数量  （分几类要设置几个阈值）

#先安装相应的包    pip install laspy<2.0.0

# import package and load data
from matplotlib.colors import Colormap


def plot_cross_section(voxel_size,input_path,dataname,height = 10,method ="log10",plot_vertical_density = False):
    
    point_cloud=lp.file.File(input_path+dataname+".las", mode="r")

    # use a vertical stack method from NumPy, and we have to transpose it to get from (n x 3) to a (3 x n) matrix of the point cloud
    points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
    #colors = np.vstack((point_cloud.red, point_cloud.green, point_cloud.blue)).transpose()

    #Point Cloud Grid Sampling 
    #The grid subsampling strategy will be based on the division of the 3D space in regular cubic cells called voxels. 
    #For each cell of this grid, we will only keep one representative point. 
    # 他的方法是基于体素的下采样  我们的是基于体素得到点云密度 从而对点云进行分类(e.g. 一个体素内点云数量>5可以仍为是数目 点云数量=1可以认为是树叶)
    # 所以我们的步骤和他一部分是类似的 不同点只在于最后的处理上

    #1. First, we create a grid structure over the points.

    #voxel_size是体素的大小
      
    #获得三个维度上体素有多少个
    #分别是 48 54 37 共9万多个点
    nb_vox=np.ceil((np.max(points, axis=0) - np.min(points, axis=0))/voxel_size)

    #2. For each small voxel, we test if it contains one or more points.
    #np.unique的用法见   https://numpy.org/doc/stable/reference/generated/numpy.unique.html
    # 下面这段代码中：
    #points - np.min(points, axis=0)) //voxel_size).astype(int) 是为了得到每个点归属于哪一个体素内
    #得到的nb_pts_per_voxel是每个体素内的点云数量
    #non_empty_voxel_keys  每一个都是三维的坐标  共11912个非空的
    #nb_pts_per_voxel  是每个非空voxel对应的点云数量
    #inverse是每个点所在的索引是多少，这里的索引仅仅针对非空的 
    non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(((points - np.min(points, axis=0)) //voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)
    #print(max(inverse))  #确定是否是对应的索引值
    map = np.zeros((int(nb_vox[0]),int(nb_vox[1])))
    need_z = height//voxel_size  #定义z值
    for idx,XYZ in enumerate(non_empty_voxel_keys):
        x,y,z = XYZ
        if z == need_z:
            if method == "log":
                map[x][y] = math.log(nb_pts_per_voxel[idx])
            elif method =="log10":
                map[x][y] = math.log10(nb_pts_per_voxel[idx])
            elif method == "original":
                map[x][y] = math.log10(nb_pts_per_voxel[idx])
    plt.imshow(map)
    plt.colorbar()
    xmax = nb_vox[0]
    ymax = nb_vox[1]
    # x_ticks = np.arange(0,xmax,5//voxel_size)
    # y_ticks = np.arange(0,ymax,5//voxel_size)
    # plt.xticks(x_ticks)
    # plt.yticks(y_ticks)
    step = 10//voxel_size 
    plt.xticks(np.arange(0,ymax,step), ['0','10','20','30','40','50'])
    plt.yticks(np.arange(0,xmax,step), ['0','10','20','30','40'])

    plt.title(dataname[:10]+" height="+str(height)+" voxel_size="+str(voxel_size))
    plt.show()   
    # 绘制垂直密度图
    #if plot_vertical_density:



if __name__ =="__main__":
    import numpy as np
    import laspy as lp
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    import math

    #input_path 你的las文件路径 如"C:/Users/Lawrence/Desktop/zsy/"
    #dataname  你的las文件名  如"zeb"，不需要后缀
    # voxel_size 所使用的体素大小
    # height 高度的水平面
    #method 取值为log或者log10或者original
    input_path="E:/dissertation/data/"
    dataname="c3_zeb_p1004_200815"
    voxel_size= 0.25 
    height = 10  
    plot_vertical_density = True
    method ="log"
    plot_cross_section(voxel_size,input_path,dataname,height,method,plot_vertical_density)


