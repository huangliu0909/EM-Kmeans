import numpy as np
import matplotlib.pyplot as plt

# 生成数据
aData = np.zeros((3, 50))
aData = (np.random.normal(0, 1, 50), np.random.normal(0, 1, 50), np.zeros(50))
bData = np.zeros((3, 50))
bData = (np.random.normal(4, 1, 50), np.random.normal(4, 1, 50), np.zeros(50))
cData = np.zeros((3, 50))
cData = (np.random.normal(0, 1, 50), np.random.normal(4, 1, 50), np.zeros(50))
dData = np.zeros((3, 50))
dData = (np.random.normal(4, 1, 50), np.random.normal(0, 1, 50), np.zeros(50))
orig_data_0 = np.concatenate((aData, bData), axis=1)
orig_data_1 = np.concatenate((cData, dData), axis=1)
orig_data = np.concatenate((orig_data_0, orig_data_1), axis=1)
t_data = (np.transpose(orig_data))
np.random.shuffle(t_data)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(aData[0], aData[1], s=30, c='b', marker='o', label='A')
ax.scatter(bData[0], bData[1],  s=30, c='r', marker='x', label='B')
ax.scatter(cData[0], cData[1], s=30, c='y', marker='o', label='C')
ax.scatter(dData[0], dData[1],  s=30, c='g', marker='x', label='D')
ax.legend()
# plt.show()
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(orig_data[0], orig_data[1],  s=30, c='k', marker='x', label='D')
ax.legend()
# plt.show()
# print(t_data)
x_0 = np.random.randint(0, 50)
x_1 = np.random.randint(0, 50)
x_2 = np.random.randint(0, 50)
x_3 = np.random.randint(0, 50)
u = np.zeros((4, 3))
u[0] = np.transpose(aData)[x_0]
u[1] = np.transpose(bData)[x_1]
u[1][2] = 1
u[2] = np.transpose(cData)[x_2]
u[2][2] = 2
u[3] = np.transpose(dData)[x_3]
u[3][2] = 3


def distance(data, center):
    # 距离函数
    a = data[0] - center[0]
    b = data[1] - center[2]
    return a * a + b * b


N = 0
while N < 500:
    # print(N)
    N += 1
    # u_old = u
    sum = np.zeros(4)
    center_x = np.zeros(4);
    center_y = np.zeros(4);
    for i in range(0, 200):
        d = np.zeros(4)
        for j in range(4):
            # 计算到每个质心的距离
            d[j] = distance(t_data[i], u[j])
        dist = np.argmin(d)
        # 取距离最近的标签
        t_data[i][2] = dist
        sum[dist] += 1
        center_x[dist] += t_data[i][0]
        center_y[dist] += t_data[i][1]
    # print(sum)
    for j in range(4):
        # 更新center
        center_x[j] = center_x[j]/sum[j]
        center_y[j] = center_y[j]/sum[j]
        u[j] = [center_x[j], center_y[j], j]

n_data_0 = np.zeros((200, 3))
n_data_1 = np.zeros((200, 3))
n_data_2 = np.zeros((200, 3))
n_data_3 = np.zeros((200, 3))
for i in range(0, 200):
    if t_data[i][2] == 0:
        n_data_0[i] = t_data[i]
    elif t_data[i][2] == 1:
        n_data_1[i] = t_data[i]
    elif t_data[i][2] == 2:
        n_data_2[i] = t_data[i]
    else:
        n_data_3[i] = t_data[i]

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(np.transpose(n_data_0)[0], np.transpose(n_data_0)[1], s=30, c='b', marker='o', label='A')
ax.scatter(np.transpose(n_data_3)[0], np.transpose(n_data_3)[1],  s=30, c='r', marker='x', label='B')
ax.scatter(np.transpose(n_data_2)[0], np.transpose(n_data_2)[1], s=30, c='y', marker='o', label='C')
ax.scatter(np.transpose(n_data_1)[0], np.transpose(n_data_1)[1],  s=30, c='g', marker='x', label='D')
ax.legend()
plt.show()