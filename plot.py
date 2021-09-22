#!/usr/bin/env python
# coding=utf-8
'''
Author: Lawrence1999
Date: 2021-09-06 17:23:55
Last Update: 2021-09-19 00:44:16
'''

# 1 用体素表示数据;
# 2 用点云表示 ;(体素大小需要测试)
# 3 上色依据点云数量  （分几类要设置几个阈值）

#先安装相应的包    pip install laspy<2.0.0

# import package and load data



def plot_cross_section(voxel_size,input_path,dataname,oritation,height ,method ="log10"):
    
    point_cloud=lp.file.File(input_path+dataname+".las", mode="r")

    # use a vertical stack method from NumPy, and we have to transpose it to get from (n x 3) to a (3 x n) matrix of the point cloud
    points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
   
    #分别是 48 54 37 共9万多个点
    nb_vox=np.ceil((np.max(points, axis=0) - np.min(points, axis=0))/voxel_size)

    non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(((points - np.min(points, axis=0)) //voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)
    #print(max(inverse))  #确定是否是对应的索引值
    print(np.max(non_empty_voxel_keys,axis = 0))
    map = [np.zeros((int(nb_vox[1])+1,int(nb_vox[2])+1)),np.zeros((int(nb_vox[0])+1,int(nb_vox[2])+1)),np.zeros((int(nb_vox[0])+1,int(nb_vox[1])+1))]
    listqqq =  ['0','10','20','30','40','50','60','70','80','90','100']
    for ori_index,ori in enumerate(oritation):
        need = height[ori_index]//voxel_size  #定义需要高度对应的体素值
        if ori == "z":
            for idx,XYZ in enumerate(non_empty_voxel_keys):
                x,y,z = XYZ
                if z == need:
                    if method == "log":
                        map[2][x][y] = math.log(nb_pts_per_voxel[idx])
                    elif method =="log10":
                        map[2][x][y] = math.log10(nb_pts_per_voxel[idx])
                    elif method == "original":
                        map[2][x][y] = math.log10(nb_pts_per_voxel[idx])
        elif ori == "x":
            for idx,XYZ in enumerate(non_empty_voxel_keys):
                x,y,z = XYZ
                if x == need:
                    if method == "log":
                        map[0][y][z] = math.log(nb_pts_per_voxel[idx])
                    elif method =="log10":
                        map[0][y][z] = math.log10(nb_pts_per_voxel[idx])
                    elif method == "original":
                        map[0][y][z] = math.log10(nb_pts_per_voxel[idx])
        elif ori == "y":
            for idx,XYZ in enumerate(non_empty_voxel_keys):
                x,y,z = XYZ
                if y == need:
                    if method == "log":
                        map[1][x][z] = math.log(nb_pts_per_voxel[idx])
                    elif method =="log10":
                        map[1][x][z] = math.log10(nb_pts_per_voxel[idx])
                    elif method == "original":
                        map[1][x][z] = math.log10(nb_pts_per_voxel[idx])                  
        if ori =="x":
            new_map = np.array(map[0]).transpose()
            plt.figure(1)
            plt.imshow(new_map)
            plt.colorbar()
            xmax = nb_vox[1]
            ymax = nb_vox[2]
            # x_ticks = np.arange(0,xmax,5//voxel_size)
            # y_ticks = np.arange(0,ymax,5//voxel_size)
            plt.xlabel("y")
            plt.ylabel("z")
            step = 10//voxel_size 
            mm = np.arange(0,xmax,step)
            plt.xticks(mm,listqqq[:mm.size])
            nn = np.arange(0,ymax,step)
            plt.yticks(nn,listqqq[:nn.size])
            plt.title('tls_1202'+" "+ori+" "+"="+str(height[ori_index])+" voxel_size="+str(voxel_size))
            plt.gca().invert_yaxis()
            plt.show()
        elif ori=="y":
            plt.figure(2)
            new_map = np.array(map[1]).transpose()
            plt.imshow(new_map)
            plt.colorbar()
            xmax = nb_vox[0]
            ymax = nb_vox[2]
            # x_ticks = np.arange(0,xmax,5//voxel_size)
            # y_ticks = np.arange(0,ymax,5//voxel_size)
            plt.xlabel("x")
            plt.ylabel("z")
            step = 10//voxel_size 
            mm = np.arange(0,xmax,step)
            plt.xticks(mm,listqqq[:mm.size])
            nn = np.arange(0,ymax,step)
            plt.yticks(nn,listqqq[:nn.size])
            plt.title('tls_1202'+" "+ori+" "+"="+str(height[ori_index])+" voxel_size="+str(voxel_size))
            plt.gca().invert_yaxis()
            plt.show()
        elif ori =="z":  
            plt.figure(3)
            new_map = np.array(map[2]).transpose()
            plt.imshow(new_map)
            plt.colorbar()
            xmax = nb_vox[0]
            ymax = nb_vox[1]
            # x_ticks = np.arange(0,xmax,5//voxel_size)
            # y_ticks = np.arange(0,ymax,5//voxel_size)
            plt.xlabel("x")
            plt.ylabel("y")
            step = 10//voxel_size 
            mm = np.arange(0,xmax,step)
            plt.xticks(mm,listqqq[:mm.size])
            nn = np.arange(0,ymax,step)
            plt.yticks(nn,listqqq[:nn.size])
            plt.title('tls_1202'+" "+ori+" "+"="+str(height[ori_index])+" voxel_size="+str(voxel_size))
            plt.gca().invert_yaxis()
            plt.show()

def plot_vertical_density(voxel_size,input_path,dataname):
    point_cloud=lp.file.File(input_path+dataname+".las", mode="r")

    # use a vertical stack method from NumPy, and we have to transpose it to get from (n x 3) to a (3 x n) matrix of the point cloud
    points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
   
    #分别是 48 54 37 共9万多个点
    nb_vox=np.ceil((np.max(points, axis=0) - np.min(points, axis=0))/voxel_size)
    listqqq =  ['0','10','20','30','40','50','60','70','80','90','100']
    non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(((points - np.min(points, axis=0)) //voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)
    # 绘制垂直密度图
    if plot_vertical_density:
        #nb_vox  是三个方向体素的大小
        map = np.zeros((int(nb_vox[0])+1,int(nb_vox[1])+1,int(nb_vox[2])+1))
        for idx,XYZ in enumerate(non_empty_voxel_keys):
            x,y,z = XYZ
            map[x][y][z]= nb_pts_per_voxel[idx]
        xlist = [i for i in range(int(nb_vox[0]))]
        ylist = [i for i in range(int(nb_vox[1]))]
        zlist = [i for i in range(int(nb_vox[2]))]
        xmean,xstd = [],[]
        ymean,ystd = [],[]
        zmean,zstd = [],[]

        for i in xlist:
            tmp = map[i][np.nonzero(map[i])]
            xmean.append(np.mean(tmp))
            xstd.append(np.std(tmp))
        xmean = np.array(xmean)
        xstd = np.array(xstd)
        # plt.scatter(xlist,xmean,marker=".")
        # step = 10//voxel_size 
        # xmax = nb_vox[0]
        # mm = np.arange(0,xmax,step)
        # plt.figure(1)
        # plt.xticks(mm,listqqq[:mm.size])
        # plt.title("x_mean")
        # plt.show()

        # plt.figure(2)
        # plt.scatter(xlist,xstd,marker=".")
        # plt.xticks(np.arange(0,xmax,step), ['0','10','20','30','40'])
        # plt.title("x_std")
        # plt.show()

        for i in ylist:
            tmp = map[:,i,:][np.nonzero(map[:,i,:])]
            ymean.append(np.mean(tmp))
            ystd.append(np.std(tmp))
        ymean = np.array(ymean)
        ystd = np.array(ystd)
        # plt.figure(3)
        # plt.scatter(ylist,ymean,marker=".")
        # # plt.scatter(ylist,ymean+ystd,marker="x")
        # # plt.scatter(ylist,ymean-ystd,marker='x')
        # xmax = nb_vox[1]
        # mm = np.arange(0,xmax,step)
        # plt.xticks(mm,listqqq[:mm.size])
        # plt.title("y_mean")
        # plt.show()

        # plt.figure(4)
        # plt.scatter(ylist,ystd,marker=".")
        # mm = np.arange(0,xmax,step)
        # plt.xticks(mm,listqqq[:mm.size])
        # plt.title("y_std")
        # plt.show()

  
        for i in zlist:
            tmp = map[:,:,i][np.nonzero(map[:,:,i])]
            zmean.append(np.mean(tmp))
            zstd.append(np.std(tmp))
        zmean = np.array(zmean)
        zstd = np.array(zstd)
        plt.figure(1)
        plt.scatter(xlist,xmean,marker=".",zorder =4)
        plt.scatter(ylist,ymean,marker=".",zorder =5)
        plt.scatter(zlist,zmean,marker=".",zorder =6)
        plt.plot(xlist,xmean,marker=".",label="x-axis",zorder =1)
        plt.plot(ylist,ymean,marker=".",label="y-axis",zorder =2)
        plt.plot(zlist,zmean,marker=".",label="z-axis",zorder =3)
        
        xmax = max(nb_vox)
        step = 10//voxel_size 
        mm = np.arange(0,xmax,step)
        plt.xticks(mm,listqqq[:mm.size])
        plt.ylabel("point density per voxel")
        plt.title("tls_1202"+"  Mean density")
        plt.legend(loc = "upper right")
        plt.show()

        plt.figure(2)
        plt.scatter(xlist,xstd,marker=".",zorder =4)
        plt.scatter(ylist,ystd,marker=".",zorder =5)
        plt.scatter(zlist,zstd,marker=".",zorder =6)
        plt.plot(xlist,xstd,marker=".",label="x-axis",zorder =1)
        plt.plot(ylist,ystd,marker=".",label="y-axis",zorder =2)
        plt.plot(zlist,zstd,marker=".",label="z-axis",zorder =3)
        xmax = max(nb_vox)
        step = 10//voxel_size 
        mm = np.arange(0,xmax,step)
        plt.xticks(mm,listqqq[:mm.size])
        plt.title("tls_1202"+"  Standard deviation of density")
        plt.ylabel("point density per voxel")
        plt.legend(loc = "upper right")
        plt.show()



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
    input_path="C:/Users/Administrator/Desktop/syz/1202/"
    dataname="tls_1202_c_nor"
    #dataname="c3_zeb_p103_200814_nor"
    voxel_size= 0.25
    # 这三个是方向,0.1,2分别表示x,y,z方向
    oritation = ["x","y","z"]
    #这三个分别是距离 这里的具体指的是从三个坐标的最小值算起 
    #height和oritaion是一一对应的
    height = [35,50,1]

    method ="log"
    plot_cross_section(voxel_size,input_path,dataname,oritation,height,method)


    plot_vertical_density(voxel_size,input_path,dataname)

