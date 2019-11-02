import numpy as np

aData = (np.random.normal(3, 1, 50))
bData = (np.random.normal(1, 1, 50))
cData = (np.random.normal(2, 1, 50))
dData = (np.random.normal(4, 1, 50))
orig_data_0 = np.concatenate((aData, bData))
orig_data_1 = np.concatenate((cData, dData))
orig_data = np.concatenate((orig_data_0, orig_data_1))
t_data = (np.transpose(orig_data))
np.random.shuffle(t_data)


def f(x, m_mu, s_sigma):
    # 计算概率密度
    a = 1./np.sqrt(2*np.pi)
    b = -(x-m_mu)**2
    c = 2*s_sigma**2
    return a * (np.exp(b / c))


gama = np.zeros((4, 200))
mu = [1, 2, 3, 4]
k = [0.25, 0.25, 0.25, 0.25]
sigma = [1, 1, 1, 1]
for s in range(5):
    for i in range(4):
        for j in range(200):
            s_sum = 0
            for t in range(4):
                s_sum += k[t]*f(t_data[j], mu[t], sigma[t])
            up = k[i]*f(t_data[j], mu[i], sigma[i])
            gama[i][j] = up / s_sum

    for i in range(4):
        # 更新 mu
        mu[i] = np.sum(gama[i]*t_data)/np.sum(gama[i])
        # 更新 sigma
        sigma[i] = np.sqrt(np.sum(gama[i] * (t_data - mu[i]) ** 2) / np.sum(gama[i]))
        # 更新 k
        k[i] = np.sum(gama[i]) / 200
error_k = np.zeros(4)
error_mu = np.zeros(4)
error_sigma = np.zeros(4)
for i in range(4):
    # 计算误差
    error_k[i] = np.abs(k[i] - 0.25) / 0.25
    error_mu[i] = np.abs(mu[i] - i-1) / (i + 1)
    error_sigma[i] = np.abs(sigma[i] - 1) /1
print("k should be [0.25, 0.25, 0.25, 0.25]")
print("actual k : " + str(k))
print("error rate : " + str(error_k))
print("mu should be [1, 2, 3, 4]")
print("actual mu : " + str(mu))
print("error rate : " + str(error_mu))
print("sigma should be [1, 1, 1, 1]")
print("actual sigma : " + str(sigma))
print("error rate : " + str(error_sigma))