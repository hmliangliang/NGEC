# -*- coding: utf-8 -*-#
# Author:       Liangliang
# Date:         2019\3\18 0008 13:57:57
# File:         NGEC.py
# Software:     PyCharm
#------------------------------------

import numpy as np
import math
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import StandardScaler
import multiprocessing

def Neighbor_similar(data):
    '''返回每个样本点的邻域相似度,数据形式为1*n,形式为:np.array([[]])'''
    #计算每个样本点的邻域相似度
    n = data.shape[0]  #数据样本的数目
    S = np.zeros((1, n)) #保存的是数据的相似度
    for i in range(n):
        f1 = 1 / (math.exp(-sum(data[i, :])) + 1)  # 计算一阶邻域相似度
        temp = np.argwhere(data[i,:]>0).transpose()#获取1-阶近邻
        value = 0
        for j in range(temp.shape[1]):
            value = value + sum(data[int(temp[0,j]),:])#计算每个邻域点的邻域点数目
        f2 = 1/(1+math.exp(-value))
        S[0,i] = 1/(1+math.exp(-(f1 + f2)))#计算每个样本点的邻域相似度
    return S


def reduction(S,d):#对应于原文中section 3.2中的降维部分
    '''
    S: 输入的数据n*n,代表一个相似度矩阵
    d: 降维后的数据的维度
    return: 返回的数据形式为n*d
    '''
    S = np.array(S)
    X = S.transpose()#将S进行转置
    P = np.random.rand(S.shape[1],d)#初始化变换矩阵P
    W = np.zeros((S.shape[0],S.shape[0]))#初始化权值矩阵W
    D = np.zeros((S.shape[0],S.shape[0]))#初始化权值度矩阵D
    for i in range(S.shape[0]):#计算权值矩阵
        for j in range(S.shape[0]):
            if i!=j:#两个不是同一个样本
                W[i,j] = math.exp(-np.linalg.norm(S[i,:]-S[j,:],2))
        value = np.mean(W[i, :])#计算均值
        for j in range(S.shape[0]):#权值低于0.8倍均值的样本之间断开连接的边
            if W[i,j]<0.8*value:
               W[i, j] = 0
        D[i, i] = sum(W[i,:])#计算度矩阵
    L = D - W#计算拉普拉斯矩阵
    values, vectors = np.linalg.eig(np.dot(np.linalg.inv(np.dot(X,np.dot(D,X.transpose()))),(np.dot(X,np.dot(L,X.transpose())) + np.eye(X.shape[0])- np.dot(X,X.transpose()))))#计算特征值
    values = np.real(values)
    vectors = np.real(vectors)
    seq = np.argsort(values)#对特征值进行排序
    seq = seq[1:d+1]
    for i in range(d):#获取最小特征值对应的特征向量
        P[:,i] = vectors[:,int(seq[i])]
    return P

def f_cluster(data,C,U,p, nu):#计算聚类的目标函数值
    value = 0
    #计算类内误差
    for i in range(data.shape[0]):#每一个样本
        for j in range(data.shape[1]):#每一个维度
            for v in range(C.shape[0]):#每一个类簇
                value = value + U[i,v]*nu[j]*(data[i,j]-C[v,j])*(data[i,j]-C[v,j])
    #计算隶属度熵
    for i in range(data.shape[0]):#每一个样本
        for v in range(C.shape[0]):  # 每一个类簇
            if U[i,v]>0:
                value = value + p*U[i,v]*math.log(U[i,v])
    return value

def Update_center(data,U,k):#更新聚类中心,k表示更新的是第k个中心点
    point = np.zeros((1,data.shape[1]))#初始化聚类中心点矩阵
    for i in range(data.shape[0]):
        point = point + U[i,k]*data[i,:]
    point = point/sum(U[:,k])
    return point[0]

def Update_membership(data,C,nu,k,p):#更新隶属度矩阵,k表示更新的是第k个样本的隶属度矩阵
    U = np.zeros((1,C.shape[0]))#初始化隶属度矩阵
    for i in range(C.shape[0]):#每个类簇
        for j in range(data.shape[1]):#每一维
            U[0,i] = U[0,i] + nu[j]*(data[k,j]-C[i,j])*(data[k,j]-C[i,j])
        U[0, i] = math.exp(- (U[0,i]+p)/p)
    return U[0]



def NGEC_cluster(data,k,P):#进行聚类
    '''
    此部分主要执行的是聚类操作
    data: 输入的数据是一个n*m的矩阵
    k: 聚类的类簇数目
    P: 降维过程的变换矩阵m*d的一个矩阵
    return: result是一个1*n的一个向量,每一个列中保存的是类标签，为np.array([[]])的格式
    '''
    p=0.001#Eq.(19)式中的rho,参数需要人为设置
    nu = np.mean(P,axis = 0)#计算每一列的均值
    if min(nu) < 0:
        nu = minmax_scale(nu)#归一化操作
    for i in range(len(nu)):
        nu[i] = nu[i]/sum(nu)
    U = np.random.rand(data.shape[0],k)#初始化隶属度矩阵
    C = np.zeros((k,data.shape[1]))#初始化中心点矩阵
    C = data[np.random.randint(0,data.shape[0],k),:]#随机地选择k个聚类中心点
    f_best = np.inf #初始化最小值
    f = f_cluster(data, C, U, p, nu)
    U_before = U#记录上一次迭代的U值
    C_before = C#记录上一次迭代的聚类中心
    N_MAX = 100#最大迭代次数
    for num in range(N_MAX):
        if f_best >= f:
            #更新聚类中心点
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())#创建一个进程池
            for i in range(k):
                C[i,:] = pool.apply_async(Update_center, (data,U,i,)).get()#使用多线程技术计算聚类的中心点
            pool.close()  # 关闭进程池，表示不能再往进程池中添加进程，需要在join之前调用
            pool.join()  # 等待进程池中的所有进程执行完毕
            #更新隶属度矩阵
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())  # 创建一个进程池
            for i in range(data.shape[0]):
                U[i,:] = pool.apply_async(Update_membership, (data,C,nu,i,p,)).get()  # 使用多进程技术计算样本的隶属度
            pool.close()  # 关闭进程池，表示不能再往进程池中添加进程，需要在join之前调用
            pool.join()  # 等待进程池中的
            f = f_cluster(data,C,U,p, nu)#计算更新后的目标函数值
            if f < f_best:#表明迭代后效果变好
                #更新聚类的参数中心点和
                f_best = f
                C_before = C
                U_before = U
            else:#表明算法的目标函数值不再下降,算法终止
                C = C_before
                U = U_before
                break
        else:#终止算法
            U = U_befor
            C = C_before
            break
    U = np.argsort(-U,axis=1)#对U进行由大到小排列
    result = U[:,0]#获取最终的聚类结果
    result = np.array([result])#将result写成1*n的np.array([[]])形式
    return result


def NGEC(data, k, d):#主函数
    '''
    data: 输入的数据,是一个n*n的矩阵,表示图的邻接矩阵,data[i,j]=1表示两个点之间存在连接
    k: 聚类的数目
    return: result为聚类的结果,为一个1*n的行向量,为np.array([[.]])形式
    '''
    n = data.shape[0]  # 数据样本的数目
    S = np.zeros((n,n))# 各个节点之间的依赖度
    for i in range(n):
        S1 = Neighbor_similar(data)#获取相似度
        for j in range(n):
            if i!=j:#不是同一个样本点
                #找出节点j及其邻域点
                tempdata = data#防止data被修改
                #将第j个节点及其邻域点之间的边删除
                tempdata[j,:] = 0
                tempdata[:,j] = 0
                S2 = Neighbor_similar(tempdata)#获取相似度
                S[i,j] = 1/(1 + math.exp(-S1[0,j]-S2[0,j]))
    #对数据进行降维处理
    P = reduction(S,d)#采用本文中的降维算法
    data = np.dot(S,P)
    data = minmax_scale(data)
    result = NGEC_cluster(data,k,P)#对数据采用文中的方法进行聚类
    return result

