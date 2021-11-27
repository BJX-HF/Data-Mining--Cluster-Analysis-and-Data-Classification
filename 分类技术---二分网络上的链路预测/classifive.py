import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

rating_names = ["user_id","movie_id","rating","timestamp"]
ratings = pd.read_table("ml-1m/ratings.dat",sep="::",header=None,names=rating_names)

#分成测试集和训练集为9:1
mask = np.random.rand(len(ratings)) < 0.9
train_ratings = ratings[mask]
test_ratings = ratings[~mask]

#将rating中的user_id和movie_id映射到对应的序号,即索引
#
users = ratings.user_id.unique()
movies = ratings.movie_id.unique()
uid2idx = {uid:k for k,uid in enumerate(users)}
mid2idx = {mid:k for k,mid in enumerate(movies)}

user_size = len(users)
movie_size = len(movies)

#建立高分选择矩阵A和选择矩阵B
A = np.zeros((user_size,movie_size))
B = np.zeros((user_size,movie_size))
for _,rating in train_ratings.iterrows():
    if(rating.rating>3): A[uid2idx[rating.user_id],mid2idx[rating.movie_id]] = 1
    B[uid2idx[rating.user_id],mid2idx[rating.movie_id]] = 1

k_user = B.sum(axis=1)
k_movie = B.sum(axis=0)
#计算W矩阵,向量加速算法
W = np.zeros((movie_size,movie_size))
A1 = A/k_user.reshape((-1,1))
A1[np.isnan(A1)] = 0
W = np.dot(A1.T,A)
W = W/k_movie
W[np.isnan(W)] = 0


#F[i][j]表示用户i下，电影j的推荐分数
# #F_sort[i][j]表示用户i下，电影j的推荐分数在所有电影中的排名
#选择A或B矩阵会得到不同的结果
F = np.dot(W,A.T).T
F_sort_index = np.argsort(F,axis=1)
F_sort = np.zeros((user_size, movie_size))
for i in range(user_size):
    for j in range(movie_size):
        F_sort[i,F_sort_index[i][j]] = movie_size - j

#准备测试集矩阵
B_test = np.zeros((user_size,movie_size))
for _,rating in test_ratings.iterrows():
    if(rating.rating>3): B_test[uid2idx[rating.user_id], mid2idx[rating.movie_id]] = 1
L = movie_size - k_user#    L:用户未选择的电影数

#计算R的值及均值
R = np.average(F_sort*B_test,axis=1)/L
r_aver = np.average(R)
print(r_aver)

#ROC曲线

TPR,FPR = [],[]
T_ = np.sum(B_test) #正样本
F_ = np.sum(B_test==False) #负样本
for threshold in np.arange(0,1,0.01):
    F_out = F_sort < (movie_size*threshold)
    F_out = F_out.astype(int)
    TP = np.sum(B_test * F_out)
    FP = np.sum((1-B_test) * F_out)
    TPR.append(TP/T_)
    FPR.append(FP/F_)
plt.plot(FPR,TPR)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel("FP")
plt.ylabel("TP")
plt.show()

#ROC曲线的积分
AUC = np.sum([ 0.01*tpr for tpr in TPR])
print(AUC)




