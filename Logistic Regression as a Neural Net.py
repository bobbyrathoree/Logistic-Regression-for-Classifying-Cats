import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage

%matplotlib inline

# To get our dataset:
def load_dataset():
    
    train_dataset = h5py.File('train.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])
    
    test_dataset = h5py.File('test.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])
    
    classes = np.array(test_dataset["list_classes"][:])
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# Load our dataset:
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Reshaping the training images:
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# Standardize our images: Divide by 255 (Maximum value of a pixel)
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

# Our Activation function: (Sigmoid)
def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

# We'd need to initialize vectors with zero and return
def init_with_zeros(size):
    weight = np.zeros(shape=(size, 1))
    bias = 0
        
    assert(weight.shape == (size, 1))
    assert(bias == 0 or isinstance(bias, float) or isinstance(bias, int))
    
    return weight, bias

# Calculate cost function and gradient (forward propogation)
def propagate(weight, bias, X, Y):
    m = X.shape[1]
    
    A = sigmoid(np.dot(weight.T, X) + bias)  # compute activation
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  # compute cost

    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)
        
    assert(dw.shape == weight.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    gradients = {"dw": dw, "db": db}
            
    return gradients, cost

# Optimize, i.e. updating weights and biases using gradient descent, minimizing cost function
def optimize(weight, bias, X, Y, alpha, number_of_iterations, print_cost=False):
    
    costs = []
    
    for i in range(number_of_iterations):
        
        gradients, cost = propagate(weight, bias, X, Y)
        
        dw = gradients["dw"]
        db = gradients["db"]
        
        weight = weight - alpha * dw
        bias = bias - alpha * db
                
        if i % 100 == 0:
            costs.append(cost)
                
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
                        
    parameters = {"w": weight, "b": bias}
    
    gradients = {"dw": dw, "db": db}
                                
    return parameters, gradients, costs

# Predict a test image:
def predict(weight, bias, X):
    
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    weight = weight.reshape(X.shape[0], 1)
    
    A = sigmoid(np.dot(weight.T, X) + bias)

    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
            
    assert(Y_prediction.shape == (1, m))
            
    return Y_prediction

# Our Logistic Regression Model:
def model(X_train, Y_train, X_test, Y_test, alpha=0.5, number_of_iterations=2000, print_cost=False):
  weight, bias = init_with_zeros(X_train.shape[0])

  parameters, gradients, costs = optimize(weight, bias, X_train, Y_train, alpha, number_of_iterations, print_cost)

  weight = parameters["w"]
  bias = parameters["b"]

  Y_prediction_test = predict(weight, bias, X_test)
  Y_prediction_train = predict(weight, bias, X_train)

  print("Train Accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
  print("Test Accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

  d = {"costs": costs,
       "Y_prediction_test": Y_prediction_test, 
       "Y_prediction_train" : Y_prediction_train, 
       "w" : weight, 
       "b" : bias,
       "learning_rate" : alpha,
       "num_iterations": number_of_iterations}

  return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, alpha = 0.005, number_of_iterations = 2000, print_cost = False)

# Let's plot our learning rate vs cost function:
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

# Test using a real unique image
my_image = "some_cat.jpg" # this is your brand new image you want to test

# We'll need to preprocess the image:
fname = my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((1, num_px * num_px * 3)).T
my_predicted_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", our algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
