# -*- coding: utf-8 -*-


import numpy as np
import laspy as lp
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# 返回三个方向的list
def count_num(input_path,data_name,size):
    point_cloud=lp.file.File(input_path+data_name+".las", mode="r")
    
    # use a vertical stack method from NumPy, and we have to transpose it to get from (n x 3) to a (3 x n) matrix of the point cloud
    points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
   
    #分别是 48 54 37 共9万多个点
    # nb_vox表示三个维度的数量
    nb_vox=np.ceil((np.max(points, axis=0) - np.min(points, axis=0))/size)

    # non_empty_voxel_keys指的是存在点云的体素索引
    # nb_pts_per_voxel指的是体素内有多少个点
    non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(((points - np.min(points, axis=0)) //size).astype(int), axis=0, return_inverse=True, return_counts=True)
    
    #定义一个list,分别用来存放三个维度的点云数量
    list_x,list_y,list_z = [0] *int(nb_vox[0]),[0] *int(nb_vox[1]),[0] *int(nb_vox[2])
    
    #保存三个维度体素数量的数组
    voxel_x,voxel_y,voxel_z = [0] *int(nb_vox[0]),[0] *int(nb_vox[1]),[0] *int(nb_vox[2])
    

    
    # 这里有两个map，map_1用来保存对应体素位置的点云数量
    # map_2只用来保存改位置是否有体素，有则为1，无则为0
    map_1 = np.zeros((int(nb_vox[0])+1,int(nb_vox[1])+1,int(nb_vox[2])+1))
    map_2 = np.zeros((int(nb_vox[0])+1,int(nb_vox[1])+1,int(nb_vox[2])+1))
    # 定义点云数据
    voxel_sum = 0
    point_sum = 0

    for idx,XYZ in enumerate(non_empty_voxel_keys):
        x,y,z = XYZ
        #map_1保存点云数量
        #map_2保存体素个数
        
        map_1[x][y][z]= nb_pts_per_voxel[idx]
        map_2[x][y][z]= 1
        voxel_sum +=1
        point_sum += nb_pts_per_voxel[idx]
        
        
    # 对三个维度先分别统计数量
    for i in range(3):
        for j in range(0,int(nb_vox[i])):
            if i==0:
                list_x[j] = np.sum(map_1[j])
                voxel_x[j] = np.sum(map_2[j])
            if i==1:
                list_y[j] = np.sum(map_1[:,j,:])
                voxel_y[j] = np.sum(map_2[:,j,:])
            if i==2:
                list_z[j] = np.sum(map_1[:,:,j])
                voxel_z[j] = np.sum(map_2[:,:,j])
                
    list_x,list_y,list_z = list_x/point_sum,list_y/point_sum,list_z/point_sum
    
    #这三个list用来表示绘图的横坐标
    xlist = [i for i in range(int(nb_vox[0]))]
    ylist = [i for i in range(int(nb_vox[1]))]
    zlist = [i for i in range(int(nb_vox[2]))]
    
    
    return xlist,ylist,zlist,list_x,list_y,list_z,voxel_x,voxel_y,voxel_z,voxel_sum




def plot_curve(input_path_tls,data_name_tls,input_path_zeb,data_name_zeb,size):
    # tls的结果
    xlist,ylist,zlist,list_x,list_y,list_z,voxel_x,voxel_y,voxel_z,voxel_num = count_num(input_path_tls,data_name_tls,size)
    
    #zeb的结果
    xlist_zeb,ylist_zeb,zlist_zeb,list_x_zeb,list_y_zeb,list_z_zeb,voxel_x_zeb,voxel_y_zeb,voxel_z_zeb,voxel_num_zeb = count_num(input_path_zeb,data_name_zeb,size)
    
    # 这里的print是用来查看两种数据剪切了之后是否是一样大
    print("xlist:",len(xlist),"ylist:",len(ylist),"zlist:",len(zlist))
    print("xlist_zeb:",len(xlist_zeb),"ylist_zeb:",len(ylist_zeb),"zlist_zeb",len(zlist_zeb))
    
    # 为了防止两者尺度不一致 取两者中的较小值为xlist
    # 并剪切list
    if (len(xlist) > len(xlist_zeb)):
        list_x = list_x[:len(list_x_zeb)]
        voxel_x = voxel_x[:len(voxel_x_zeb)]
        xlist = xlist_zeb
    else:
        list_x_zeb = list_x_zeb[:len(list_x)]
        voxel_x_zeb = voxel_x_zeb[:len(voxel_x)]
        
    if (len(ylist) > len(ylist_zeb)):
        list_y = list_y[:len(list_y_zeb)]
        voxel_y = voxel_y[:len(voxel_y_zeb)]
        ylist = ylist_zeb
    else:
        list_y_zeb = list_y_zeb[:len(list_y)]
        voxel_y_zeb = voxel_y_zeb[:len(voxel_y)]
        
    if (len(zlist) > len(zlist_zeb)):
        list_z = list_z[:len(list_z_zeb)]
        voxel_z = voxel_z[:len(voxel_z_zeb)]
        zlist = zlist_zeb
    else:
        list_z_zeb = list_z_zeb[:len(list_z)]
        voxel_z_zeb = voxel_z_zeb[:len(voxel_z)]
    #print("xlist:",len(list_x),"ylist:",len(list_y),"zlist:",len(list_z))
    #print("xlist_zeb:",len(list_x_zeb),"ylist_zeb:",len(list_y_zeb),"zlist_zeb",len(list_z_zeb))
    #list_x_zeb,list_y_zeb,list_z_zeb = list_x_zeb[:len(list_x)],list_y_zeb[:len(list_y)],list_z_zeb[:len(list_z)]
    

    xlist,ylist,zlist = list(np.array(xlist)*size),list(np.array(ylist)*size),list(np.array(zlist)*size)
    

    
    # 方法二的图
    figure = plt.figure(figsize = (4,13))

    
    figure5 = plt.subplot(311)
    plt.scatter(xlist,voxel_x,marker=".",zorder =3)
    plt.scatter(xlist,voxel_x_zeb,marker=".",zorder =4)
    plt.plot(xlist,voxel_x,marker=".",label="TLS",zorder =1)
    plt.plot(xlist,voxel_x_zeb,marker=".",label="ZEB",zorder =2)
    plt.xlabel("Direction of X axis(m)")
    plt.ylabel("Number of voxels")
    plt.legend(loc = "upper right")
    plt.suptitle("voxel_size =" +str(size) +"m",y=0.9,fontsize =16)


    
    figure6 = plt.subplot(312)
    plt.scatter(ylist,voxel_y,marker=".",zorder =3)
    plt.scatter(ylist,voxel_y_zeb,marker=".",zorder =4)
    plt.plot(ylist,voxel_y,marker=".",label="TLS",zorder =1)
    plt.plot(ylist,voxel_y_zeb,marker=".",label="ZEB",zorder =2)
    plt.xlabel("Direction of Y axis(m)")
    plt.ylabel("Number of voxels")
    plt.legend(loc = "upper right")

    figure7 = plt.subplot(313)
    plt.scatter(zlist,voxel_z,marker=".",zorder =3)
    plt.scatter(zlist,voxel_z_zeb,marker=".",zorder =4)
    plt.plot(zlist,voxel_z,marker=".",label="TLS",zorder =1)
    plt.plot(zlist,voxel_z_zeb,marker=".",label="ZEB",zorder =2)
    plt.xlabel("Direction of Z axis(m)")
    plt.ylabel("Number of voxels")
    plt.legend(loc = "upper right")
    plt.show()
    

    figure = plt.figure(figsize = (4,13))

    figure1 = plt.subplot(311)
    plt.scatter(xlist,list_x,marker=".",zorder =3)
    plt.scatter(xlist,list_x_zeb,marker=".",zorder =4)
    plt.plot(xlist,list_x,marker=".",label="TLS",zorder =1)
    plt.plot(xlist,list_x_zeb,marker=".",label="ZEB",zorder =2)
    plt.xlabel("Direction of X axis(m)")
    plt.ylabel("Percentage of points(%)")
    plt.legend(loc = "upper right")

    figure2 = plt.subplot(312)
    plt.scatter(ylist,list_y,marker=".",zorder =3)
    plt.scatter(ylist,list_y_zeb,marker=".",zorder =4)
    plt.plot(ylist,list_y,marker=".",label="TLS",zorder =1)
    plt.plot(ylist,list_y_zeb,marker=".",label="ZEB",zorder =2)
    plt.xlabel("Direction of Y axis(m)")
    plt.ylabel("Percentage of points(%)")
    plt.legend(loc = "upper right")
    plt.suptitle("voxel_size =" +str(size) +"m",y=0.9,fontsize =16)
    
    
    figure3 = plt.subplot(313)
    plt.scatter(zlist,list_z,marker=".",zorder =3)
    plt.scatter(zlist,list_z_zeb,marker=".",zorder =4)
    plt.plot(zlist,list_z,marker=".",label="TLS",zorder =1)
    plt.plot(zlist,list_z_zeb,marker=".",label="ZEB",zorder =2)
    plt.xlabel("Direction of Z axis(m)")
    plt.ylabel("Percentage of points(%)")
    plt.legend(loc = "upper right")
    plt.show()
    
    
   
 
    
    # 方法2的统计量
    r2_x = r2_score(list_x,list_x_zeb)
    r2_y = r2_score(list_y,list_y_zeb)
    r2_z = r2_score(list_z,list_z_zeb)
    
    rmse_x = np.sqrt(mean_squared_error(list_x,list_x_zeb))
    rmse_y = np.sqrt(mean_squared_error(list_y,list_y_zeb))
    rmse_z = np.sqrt(mean_squared_error(list_z,list_z_zeb))
    
    # 方法1的统计量
    voxel_x_percent,voxel_y_percent,voxel_z_percent = list(np.array(voxel_x)/voxel_num),list(np.array(voxel_y)/voxel_num),list(np.array(voxel_z)/voxel_num),
    
    voxel_x_zeb_percent, voxel_y_zeb_percent, voxel_z_zeb_percent = list(np.array(voxel_x_zeb)/voxel_num_zeb),list(np.array(voxel_y_zeb)/voxel_num_zeb),list(np.array(voxel_z_zeb)/voxel_num_zeb)
    
    r2_x_voxel = r2_score(voxel_x_percent,voxel_x_zeb_percent)
    r2_y_voxel = r2_score(voxel_y_percent,voxel_y_zeb_percent)
    r2_z_voxel = r2_score(voxel_z_percent,voxel_z_zeb_percent)
    
    rmse_x_voxel = np.sqrt(mean_squared_error(voxel_x_percent,voxel_x_zeb_percent))
    rmse_y_voxel = np.sqrt(mean_squared_error(voxel_y_percent,voxel_y_zeb_percent))
    rmse_z_voxel = np.sqrt(mean_squared_error(voxel_z_percent,voxel_z_zeb_percent))
    
    
    
    print("voxel_size = " +str(size))
    
    print("方法1（各维度体素数量）的相关统计量")
    print("r2_x_voxel =" +str(r2_x_voxel))
    print("r2_y_voxel =" +str(r2_y_voxel))
    print("r2_z_voxel =" +str(r2_z_voxel))
    
    print("RMSE_x_voxel="+str(rmse_x_voxel))
    print("RMSE_y_voxel="+str(rmse_y_voxel))
    print("RMSE_z_voxel="+str(rmse_z_voxel))
    
    print("\n方法2（各维度百分比）的相关统计量")
    print("r2_x =" +str(r2_x))
    print("r2_y =" +str(r2_y))
    print("r2_z =" +str(r2_z))
    
    print("RMSE_x="+str(rmse_x))
    print("RMSE_y="+str(rmse_y))
    print("RMSE_z="+str(rmse_z))
    
    
    
    
if __name__ =="__main__":
    
    input_path_tls = r"C:/Users/74130/Desktop/张诗雨论文/"
    data_name_tls = "tls_1202_c_nor"
    
    input_path_zeb = r"C:/Users/74130/Desktop/张诗雨论文/"
    data_name_zeb = "zeb_p1202_nor"
    
    plot_curve(input_path_tls,data_name_tls,input_path_zeb,data_name_zeb,1)
    