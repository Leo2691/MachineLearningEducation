import numpy as np

def convolution(X, h, p = False):

    Y = np.array(np.zeros(len(X)), dtype=complex)

    if p == True:
        for i in np.arange(len(X)):
            for j in np.arange(len(h)):
                Y[i] += h[j] * np.power(abs(X[i]), j)

    else:
        for i in np.arange(len(h) - 1, len(X)):
            for j in np.arange(len(h)):
                Y[i] += h[j] * X[i - j]


    return Y



def Gradient(D, X, h0, h1, h2):

    i = complex(0, 1)

    y = np.array(np.zeros(len(X)), dtype=complex)
    e = np.array(np.zeros(len(X)), dtype=complex)
    error_ = np.array(np.zeros(len(X)), dtype=complex)
    delta_h0 = np.array(np.zeros(len(X)), dtype=complex)


    for k in np.arange(len(h0), len(X) - 1):
        mult = convolution(X[k - len(h0):k], h0, False) * convolution(convolution(X[k - len(h2):k], h2, True), h1, False)
        y[k - 1] = mult[len(h0) - 1]

        e[k - 1] = D[k - 1] - y[k - 1]
        error_[k - 1] = e[k - 1] * np.conj(e[k - 1])

        s = convolution(convolution(X[k - len(h2):k], h2, True), h1, False)

        learaning_rate_h0 = 0.0001
        for n in np.arange(len(h0)):
            #delta = s * learaning_rate_h0 * X[k - n - 1]
            #delta = delta[len(delta) - 1]

            #real_h0 = e[k - 1] * np.conj(delta) + delta * np.conj(e[k - 1])
            #imag_h0 = e[k - 1] * i * np.conj(delta) - i * delta * np.conj(e[k - 1])

            #corr = real_h0 * learaning_rate_h0 + i * imag_h0 * learaning_rate_h0

            delta = s * learaning_rate_h0 * X[k - n - 1]

            h0[n] = h0[n] - delta[len(delta) - 1]

            print(1)

        print(1)


    print (1)


h0 = np.array([complex(0.1, 0.2), complex(0.2, -0.4), complex(0.4, 0.6)])
h2 = np.array([complex(0.2, 0.4), complex(0.3, -0.1), complex(0.6, 0.2)])
h1 = np.array([complex(0.1, -0.3), complex(0.2, 0.8), complex(0.5, 0.9)])
X = np.array([complex(2, 5), complex(3, -15), complex(6, 3), complex(8, 12), complex(17, 5), complex(5, -4), complex(1, 2), complex(3, 5), complex(1, 3)])

X_h0 = convolution(X, h0, False)
X_h2 = convolution(X, h2, True)
X_h1 = convolution(X_h2, h1, False)

Y = X_h0 * X_h1


h0_ = np.array([complex(0.11, 0.19), complex(0.18, -0.36), complex(0.36, 0.59)])
Gradient(Y, X, h0_, h1, h2)

print(1)