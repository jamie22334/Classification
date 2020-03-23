import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2

    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0

    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0

        w2 = np.insert(w, 0, b)
        X2 = np.insert(X, [0], 1, axis=1)
        y2 = np.where(y != 0, y, -1)
        # print("y label:" + np.array_str(y))

        # print("perceptron: ")
        # print("w shape:" + str(w2.shape))
        # print("X shape:" + str(X2.shape))
        # print("y shape:" + str(y2.shape))

        for iter in range(max_iterations):
            ids = np.where(y2 * np.dot(w2, X2.T) <= 0)
            sum_vector = np.dot(y2[ids], X2[ids])
            w2 = w2 + sum_vector
        w2 = np.dot(step_size / N, w2)

        b = w2[0]
        w = w2[1:]
        ############################################

    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0

        w2 = np.insert(w, 0, b)
        X2 = np.insert(X, [0], 1, axis=1)
        y2 = np.where(y != 0, y, -1)

        # print("logistic: ")
        # print("w shape:" + str(w2.shape))
        # print("X shape:" + str(X2.shape))
        # print("y shape:" + str(y2.shape))

        # val1 = np.dot(w2, X2.T)
        # val2 = np.multiply(-y2, np.dot(w2, X2.T))
        # val3 = sigmoid(np.multiply(-y2, np.dot(w2, X2.T)))
        # val4 = np.multiply(sigmoid(np.multiply(-y2, np.dot(w2, X2.T))), y2)
        # val5 = np.dot(np.multiply(sigmoid(np.multiply(-y2, np.dot(w2, X2.T))), y2), X2)
        # print("shape1: " + str(val1.shape))
        # print("shape2: " + str(val2.shape))
        # print("shape3: " + str(val3.shape))
        # print("shape4: " + str(val4.shape))
        # print("shape5: " + str(val5.shape))
        for iter in range(max_iterations):
            # lecture SGD
            sum_vector = np.dot(np.multiply(sigmoid(np.multiply(-y2, np.dot(w2, X2.T))), y2), X2)
            w2 = w2 + np.dot(step_size / N, sum_vector)

        b = w2[0]
        w = w2[1:]
        ############################################

    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    return w, b


def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    value = 1 / (1 + np.exp(-z))
    ############################################
    
    return value


def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    
    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)

        for i in range(N):
            if np.dot(w.T, X[i]) + b > 0:
                preds[i] = 1
            else:
                preds[i] = 0
        ############################################

    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)

        for i in range(N):
            if sigmoid(np.dot(w.T, X[i]) + b) > 0.5:
                preds[i] = 1
            else:
                preds[i] = 0
        ############################################

    else:
        raise "Loss Function is undefined."

    assert preds.shape == (N,) 
    return preds


def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)

        w = np.insert(w, 0, b, axis=1)
        X = np.insert(X, [0], 1, axis=1)
        # y_matrix = np.zeros(C * N).reshape(N, C)
        #
        # for i in range(N):
        #     y_matrix[i][y[i]] = 1

        for iter in range(max_iterations):
            random_n = np.random.choice(N)
            class_arr = np.dot(w, X[random_n].T)
            class_arr = class_arr - np.max(class_arr)
            class_exp = np.exp(class_arr)
            sum_exp = np.sum(class_exp)
            softmax_arr = np.divide(class_exp, sum_exp)
            softmax_arr[y[random_n]] = softmax_arr[y[random_n]] - 1

            sum_matrix = np.dot(step_size, np.dot(np.array([softmax_arr]).T, np.array([X[random_n]])))
            w = w - sum_matrix

        b = w[0:, 0]
        w = w[0:, 1:]
        ############################################

    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)

        w2 = np.insert(w, 0, b, axis=1)
        X2 = np.insert(X, [0], 1, axis=1)
        y_matrix = np.zeros(C * N).reshape(C, N)

        for i in range(N):
            y_matrix[y[i]][i] = 1

        # for iter in range(max_iterations):
        #     class_arr = np.dot(w2, X2.T)
        #     class_arr = class_arr - np.max(class_arr)
        #     class_exp = np.exp(class_arr)
        #     sum_exp = np.sum(class_exp, axis=0)
        #     softmax_arr = np.divide(class_exp, sum_exp)
        #
        #     softmax_arr = softmax_arr - y_matrix
        #
        #     sum_matrix = np.dot(step_size / N, np.dot(softmax_arr, X2))
        #     w2 = w2 - sum_matrix

        b = w2[0:, 0]
        w = w2[0:, 1:]
        ############################################

    else:
        raise "Type of Gradient Descent is undefined."

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D 
    - b: bias terms of the trained multinomial classifier, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    preds = np.zeros(N)

    for i in range(N):
        score = np.add(np.dot(w, X[i].T), b)
        classification = np.argmax(score)
        preds[i] = classification
    # print("predict: " + np.array_str(preds))
    ############################################

    assert preds.shape == (N,)
    return preds
