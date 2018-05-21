# Back-Propagation Neural Networks
#
import numpy as np
import matplotlib.pyplot as plt


# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return np.tanh(x)
    # return 1 / (1+np.exp(-x))


# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2
    # return y*(1-y)


class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1  # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = np.ones((self.ni, ))
        self.ah = np.ones((self.nh, ))
        self.ao = np.ones((self.no, ))

        # create weights
        # self.wi = makeMatrix(self.ni, self.nh)
        # self.wo = makeMatrix(self.nh, self.no)

        # set them to random vaules
        self.wi = np.random.uniform(
            low=-0.2, high=0.2, size=(self.ni, self.nh))
        self.wo = np.random.uniform(low=-2, high=2, size=(self.nh, self.no))

        # last change in weights for momentum
        self.ci = np.zeros((self.ni, self.nh))
        self.co = np.zeros((self.nh, self.no))

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        self.ai[:-1] = inputs
        self.ah = sigmoid(np.dot(self.ai, self.wi))
        self.ao = sigmoid(np.dot(self.ah, self.wo))

        return self.ao

    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        output_deltas = dsigmoid(self.ao) * (targets - self.ao)

        hidden_deltas = dsigmoid(self.ah) * np.dot(self.wo, output_deltas)

        change = np.dot(self.ah.reshape((-1, 1)),
                        output_deltas.reshape((-1, 1)).T)
        self.wo = self.wo + N * change + M*self.co
        self.co = change

        change = np.dot(self.ai.reshape((-1, 1)),
                        hidden_deltas.reshape((-1, 1)).T)
        self.wi = self.wi + N * change + M*self.ci
        self.ci = change

        # calculate error
        diff = (targets - self.ao)
        error = np.dot(diff.T, diff)

        return error

    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]))

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=1000, N=0.5, M=0.1):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            if i % 100 == 0:
                print('error %-.5f' % error)


class NN_GEN:
    def __init__(self, Ns):
        assert len(Ns) > 1
        # number of input, hidden, and output nodes
        self.Ns = Ns
        self.Ns[0] += 1
        self.layers = len(self.Ns)

        self.As = []
        for n in self.Ns:
            self.As.append(np.ones((n, )))

        self.W = []
        for ni, no in zip(self.Ns[:-1], self.Ns[1:]):
            self.W.append(np.random.uniform(low=-1, high=1, size=(ni, no)))

        self.C = []
        for ni, no in zip(self.Ns[:-1], self.Ns[1:]):
            self.C.append(np.zeros((ni, no)))

    def update(self, inputs):
        if len(inputs) != self.Ns[0]-1:
            raise ValueError('wrong number of inputs')

        # input activations
        self.As[0][:-1] = inputs
        for i in range(self.layers - 1):
            self.As[i + 1] = sigmoid(np.dot(self.As[i], self.W[i]))

        return self.As[-1]

    def backPropagate(self, targets, N, M):
        if len(targets) != self.Ns[-1]:
            raise ValueError('wrong number of target values')

        deltas = dsigmoid(self.As[-1]) * (targets - self.As[-1])
        for i in range(self.layers-1, 0, -1):
            change = np.dot(self.As[i-1].reshape((-1, 1)),
                            deltas.reshape((-1, 1)).T)
            self.W[i-1] = self.W[i-1] + N * change + M*self.C[i-1]
            self.C[i-1] = change
            deltas = dsigmoid(self.As[i-1]) * np.dot(self.W[i-1], deltas)

        # calculate error
        diff = (targets - self.As[-1])
        error = np.dot(diff.T, diff)

        return error

    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]))

    def weights(self):
        print('weights:')
        for w in self.W:
            print(w)

    def train(self, patterns, iterations=1000, N=0.5, M=0.1):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            if i % 100 == 0:
                print('error %-.5f' % error)


def demo():
    X = np.array([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]])

    Y = np.array([[1.],
                  [0.],
                  [0.],
                  [1.]])

    pat = list(zip(X, Y))

    # create a network with two input, two hidden, and one output nodes
    n = NN_GEN([2, 2, 1])
    # train it with some patterns
    n.train(pat)
    # test it
    n.test(pat)


def plotting(min=0, max=1, l=500, iter=100):
    X = np.concatenate((np.random.uniform(min, max, size=(l, 2)), np.ones((l, 1))), axis = 1)
    wc = np.array([0.5, 1, -1])
    y = (np.dot(X, wc) > 0) * 2 - 1
    alpha = alpha0 = 0.05
    w = np.random.randn(3)

    for i in range(iter):
        c = (np.dot(X, w) > 0) != (y > 0)
        if sum(c) == 0:
            break
        w += alpha * np.dot(y[c] - 0, X[c])
        #w /= w[1]
        if i > 50:
            alpha = alpha0/(i-50)
        plt.clf()

        plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'b.',
                 X[:, 0][y == -1], X[:, 1][y == -1], 'r.',
                 X[:, 0], -w[2]/w[1]-w[0]/w[1]*X[:, 0],
                 X[:, 0], -wc[2]/wc[1]-wc[0]/wc[1]*X[:, 0])
        plt.xlim((min, max))
        plt.ylim((min, max))
        plt.draw()
        plt.pause(.1)
    
    print(i, sum(c))


if __name__ == '__main__':
    demo()
    pass
