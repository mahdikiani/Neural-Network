import matplotlib.pyplot as plt
import numpy as np

n = 10
epochs = 300
eta = 0.1

cities = np.random.rand(n, 2)

numNodes = 2 * n
nodes = np.random.rand(numNodes, 2)


def neighborhoodGaussian(nearest, nodeIdx, epoch, numNodes):
    distanceClockwise = np.abs(nearest - nodeIdx)
    distanceAntiClockwise = numNodes - distanceClockwise
    distance = min(distanceClockwise, distanceAntiClockwise)
    # if distance == 0:
    #     return 1
    # if distance <= 1:
    #     return 0.5 / epoch
    # return 0
    return np.exp(-((distance * 1.0 * epoch / 100) ** 2) / 2)


def epoch(num):
    np.random.shuffle(cities)
    for city in cities:
        nearest = np.argmin(np.linalg.norm(nodes - city, 2, axis=1))
        for nodeIdx in range(nodes.shape[0]):
            theta = neighborhoodGaussian(nearest, nodeIdx, num+1, numNodes)
            nodes[nodeIdx] += theta * eta * (city - nodes[nodeIdx])


plt.plot(cities[:, 0], cities[:, 1], 'b.')
plt.show()

show = 1

for e in range(epochs):
    epoch(e)
    if show and e%10 == 0:
        path = np.concatenate((nodes, nodes[0].reshape(-1, 1).T))
        l = plt.plot(path[:, 0], path[:, 1], 'r')
        plt.setp(l, linewidth=.5)
        plt.plot(nodes[:, 0], nodes[:, 1], 'r.')
        plt.plot(cities[:, 0], cities[:, 1], 'b.')

        # plt.savefig('res/' + str(e) + '.jpg')

        plt.pause(.005)
        plt.clf()

nodes = np.concatenate((nodes, nodes[0].reshape(-1, 1).T))
l2 = plt.plot(nodes[:, 0], nodes[:, 1], 'r')
plt.setp(l2, linewidth=.5)
plt.plot(nodes[:, 0], nodes[:, 1], 'r.')
plt.plot(cities[:, 0], cities[:, 1], 'b.')
plt.show()
