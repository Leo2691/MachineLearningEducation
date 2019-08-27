import numpy as np

def generate_signal(X):
    length_window = 0

    #additional fiter
    h0 = np.array([complex(0.1, 0.2), complex(0.2, -0.4), complex(0.4, 0.6)])
    X_add_h0 = np.array(np.zeros(len(X)), dtype = complex)
    for i in np.arange(len(h0) - 1, len(X)):
        for j in np.arange(len(h0)):
            X_add_h0[i] += h0[j] * X[i - j]


    h2 = np.array([complex(0.2, 0.4), complex(0.3, -0.1), complex(0.6, 0.2)])
    P = len(h2)
    X_pow = np.array(np.zeros(len(X)), dtype = complex)
    for i in np.arange(len(X)):
        for p in np.arange(P):
            X_pow[i] += h2[p] * pow(abs(X[i]), p)


    h1 = np.array([complex(0.1, -0.3), complex(0.2, 0.8), complex(0.5, 0.9)])
    X_pow_add_h1 = np.array(np.zeros(len(X)), dtype = complex)
    for i in np.arange(len(h0) - 1, len(X)):
        for j in np.arange(len(h1)):
            X_pow_add_h1[i] += h1[j] * X_pow[i - j]


    Y = X_add_h0 * X_pow_add_h1
    print(X, '\n\n', X_add_h0, '\n\n', X_pow_add_h1, '\n\n', Y, '\n\n')

    return Y

# alghoritm

def y_predict(X, length_window, P, h0, h1, h2):

    Y_pred = np.array(np.zeros(len(X)), dtype=complex)

    # additional fiter
    #h0 = complex(0, 0)
    X_add_h0 = np.array(np.zeros(len(X)), dtype=complex)
    for i in np.arange(length_window - 1, len(X)):
        for j in np.arange(length_window):
            X_add_h0[i] += h0[j] * X[i - j]

    #h2 = complex(0, 0)
    #P = 4
    X_pow = np.array(np.zeros(len(X)), dtype=complex)
    for i in np.arange(len(X)):
        for p in np.arange(P):
            X_pow[i] += h2[p] * pow(abs(X[i]), p)

    #h1 =  complex(0, 0)
    X_pow_add_h1 = np.array(np.zeros(len(X)), dtype=complex)
    for i in np.arange(length_window - 1, len(X)):
        for j in np.arange(length_window):
            X_pow_add_h1[i] += h1[j] * (X_pow[i - j])

    Y_pred = X_add_h0 * X_pow_add_h1

    return Y_pred

def h0_derivative(X, length_window, P, h1, h2):

    Y_pred = np.array(np.zeros(len(X)), dtype=complex)

    #h2 = complex(0, 0)
    #P = 4
    X_pow = np.array(np.zeros(len(X)), dtype=complex)
    for i in np.arange(len(X)):
        for p in np.arange(P):
            X_pow[i] += h2[p] * pow(abs(X[i]), p)

    #h1 =  complex(0, 0)
    X_pow_add_h1 = complex(0, 0)
    #for i in np.arange(len(X) - 1, -1, -1):
    i = len(X) - 1
    for j in np.arange(length_window):
        X_pow_add_h1 += h1[j] * (X_pow[i - j])

    Y_pred = X_pow_add_h1

    return -Y_pred

def h2_derivative(X, length_window, P, h0, h1):
    Y_pred = np.array(np.zeros(len(X)), dtype=complex)

    # additional fiter
    X_add_h0 = complex(0, 0)
    i = len(X) - 1
    for j in np.arange(length_window):
        X_add_h0 += h0[j] * X[i - j]

    # h2 = complex(0, 0)
    # P = 4
    X_pow = np.array(np.zeros(len(X)), dtype=complex)
    for i in np.arange(len(X)):
        for p in np.arange(P):
            X_pow[i] += pow(abs(X[i]), p)

    # h1 =  complex(0, 0)
    X_pow_add_h1 = complex(0, 0)
    i = len(X) - 1
    for j in np.arange(length_window):
        X_pow_add_h1 += h1[j] * (X_pow[i - j])

    Y_pred = X_add_h0 * X_pow_add_h1

    return -Y_pred

def h1_derivative(X, length_window, P, h0, h2):

    Y_pred = np.array(np.zeros(len(X)), dtype=complex)

    # additional fiter
    #h0 = complex(0, 0)
    X_add_h0 = complex(0, 0)
    i = len(X) - 1
    for j in np.arange(length_window):
        X_add_h0 += h0[j] * X[i - j]

    #h2 = complex(0, 0)
    #P = 4
    X_pow = np.array(np.zeros(len(X)), dtype=complex)
    for i in np.arange(len(X)):
        for p in np.arange(P):
            X_pow += h2[p] * pow(abs(X[i]), p)

    """"""
    Y_pred = X_add_h0 * X_pow

    return Y_pred

# error squared
def Error(y, X, N, P, h0, h1, h2):

    error = np.sum(y - y_predict(X, N, P, h0, h1, h2))

    return error#* complex.conjugate(error)


def Gradient(y, X, N, P, h0, h1, h2, nu = .001, mu = .000001, tetta = .00001, printing = True):

    for i in np.arange(10000):

        Errors = []

        for i in np.arange(len(X)):

            Errors.append(Error(y[i:i+N], X[i:i+N], N, P, h0, h1, h2))
            print(Errors)

            h0_re = np.array(h0.real)
            h0_im = np.array(h0.imag)

            for n in np.arange(len(h0)):
                error = Error(y[i:i+N], X[i:i+N], N, P, h0, h1, h2)
                error_= complex.conjugate(Error(y[i:i+N], X[i:i+N], N, P, h0, h1, h2))
                grad_ = complex.conjugate(h0_derivative(X[i:i+N], N, P, h1, h2))
                grad =  h0_derivative(X[i:i+N], N, P, h1, h2)


                im_unit = complex(0, 1)
                Re =  (error * -grad_ + -grad * error_).real
                Im = (error * -im_unit * grad_ + im_unit * grad * error_).real

                h0[n] = complex(h0[n].real - nu * (Re), h0[n].imag - nu * (Im))
                #h0[n].imag -= learning_rate * (Im)

            for n in np.arange(len(h2)):
                error = Error(y[i:i + N], X[i:i + N], N, P, h0, h1, h2)
                error_ = complex.conjugate(Error(y[i:i + N], X[i:i + N], N, P, h0, h1, h2))
                grad_ = complex.conjugate(h2_derivative(X[i:i + N], N, P, h0, h1))
                grad = h2_derivative(X[i:i + N], N, P, h0, h1)

                im_unit = complex(0, 1)
                Re = (error * -grad_ + -grad * error_).real
                Im = (error * -im_unit * grad_ + im_unit * grad * error_).real

                h2[n] = complex(h2[n].real - mu * (Re), h2[n].imag - mu * (Im))
                # h0[n].imag -= learning_rate * (Im)

            for n in np.arange(len(h1)):
                error = Error(y[i:i + N], X[i:i + N], N, P, h0, h1, h2)
                error_ = complex.conjugate(Error(y[i:i + N], X[i:i + N], N, P, h0, h1, h2))
                grad_ = complex.conjugate(h1_derivative(X[i:i + N], N, P, h0, h2))
                grad = h1_derivative(X[i:i + N], N, P, h0, h2)

                im_unit = complex(0, 1)
                Re = (error * -grad_ + -grad * error_).real
                Im = (error * -im_unit * grad_ + im_unit * grad * error_).real

                h1[n] = complex(h1[n].real - tetta * (Re), h1[n].imag - mu * (Im))
                # h0[n].imag -= learning_rate * (Im)


                if (printing):
                    print("Iteration {0}: MSE = {1:.6f}, h0 = {2:.6f}, h1 = {3:.6f}, h2 = {4:.6f}".format(i, Errors[i], h0, h1, h2))


#print(X.shape, d.shape)


X = np.array([complex(2, 5), complex(3, -15), complex(6, 3), complex(8, 12), complex(17, 5), complex(5, -4), complex(1, 2)])
Y = generate_signal(X)
h0 = np.array([complex(0.1, 0.2), complex(0.2, 0.1), complex(0.1, 0.2)])
h1 = np.array([complex(0.1, 0.1), complex(0.1, 0.2), complex(0.1, 0.3)])
h2 = np.array([complex(0.1, -0.2), complex(0.1, 0.2), complex(-0.4, 0.3)])


Y_pred  = y_predict(X, 3, 3, h0, h1, h2)

print('\n\n',  X, '\n\n', Y, '\n\n', Y_pred)

Gradient(Y, X, 3, 3, h0, h1, h2, nu = .00001, mu = .0000001, tetta = .0000001, printing = True)

