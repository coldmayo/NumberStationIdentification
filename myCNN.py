import numpy as np
import pandas as pd

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=0, keepdims=True)

def ReLU(A):
    return [max(0,i) for i in A]

def one_hot(y):
    y = pd.Series(y)
    y = pd.get_dummies(y).values.tolist()
    return np.array(y)

class ConvolutionLayer:
    def __init__(self, kernel_num, kernel_size):
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        # make random filters to start
        self.kernels = np.random.randn(kernel_num, kernel_size, kernel_size) / (kernel_size**2)

    def patches_generator(self, image):
        # Extract image height and width
        hei, wei = image.shape
        self.image = image
        # The number of patches, given a fxf filter is h-f+1 for height and w-f+1 for width
        for h in range(hei-self.kernel_size+1):
            for w in range(wei-self.kernel_size+1):
                patch = image[h:(h+self.kernel_size), w:(w+self.kernel_size)]
                yield patch, h, w   # yield is return but for iteration
    
    def forward_prop(self, image):
        hei, wei = image.shape
        # Initialize the convolution output volume of the correct size
        convolution_output = np.zeros((hei-self.kernel_size+1, wei-self.kernel_size+1, self.kernel_num))
        # Unpack the generator
        for patch, h, w in self.patches_generator(image):
            # Perform convolution for each patch
            convolution_output[h,w] = np.sum(patch*self.kernels, axis=(1,2))
        return convolution_output
    
    def back_prop(self, dE_dY, alpha):
        # Initialize gradient of the loss function with respect to the kernel weights
        dE_dk = np.zeros(self.kernels.shape)
        for patch, h, w in self.patches_generator(self.image):
            for f in range(self.kernel_num):
                dE_dk[f] += patch * dE_dY[h, w, f]
        # Update the parameters
        self.kernels -= alpha*dE_dk
        return dE_dk

class MaxPoolingLayer:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def patches_generator(self, image):
        # Compute the ouput size
        output_h = int(image.shape[0]/self.kernel_size)
        output_w = int(image.shape[1]/self.kernel_size)
        self.image = image

        for h in range(output_h):
            for w in range(output_w):
                patch = image[(h*self.kernel_size):(h*self.kernel_size+self.kernel_size), (w*self.kernel_size):(w*self.kernel_size+self.kernel_size)]
                yield patch, h, w

    def forward_prop(self, image):
        image_h, image_w, num_kernels = image.shape
        max_pooling_output = np.zeros((int(image_h/self.kernel_size), int(image_w/self.kernel_size), num_kernels))
        for patch, h, w in self.patches_generator(image):
            max_pooling_output[h,w] = np.amax(patch, axis=(0,1))
        return max_pooling_output

    def back_prop(self, dE_dY):
        dE_dk = np.zeros(self.image.shape)
        for patch,h,w in self.patches_generator(self.image):
            image_h, image_w, num_kernels = patch.shape
            max_val = np.amax(patch, axis=(0,1))

            for idx_h in range(image_h):
                for idx_w in range(image_w):
                    for idx_k in range(num_kernels):
                        if patch[idx_h,idx_w,idx_k] == max_val[idx_k]:
                            dE_dk[h*self.kernel_size+idx_h, w*self.kernel_size+idx_w, idx_k] = dE_dY[h,w,idx_k]
            return dE_dk

class SoftmaxLayer:
    def __init__(self, input_units, output_units):
        # Initiallize weights and biases
        self.weight = np.random.randn(input_units, output_units)/input_units
        self.bias = np.zeros(output_units)

    def forward_prop(self, image):
        self.original_shape = image.shape # stored for backprop
        # Flatten the image
        image_flattened = image.flatten()
        self.flattened_input = image_flattened # stored for backprop
        # Perform matrix multiplication and add bias
        first_output = np.dot(image_flattened, self.weight) + self.bias
        self.output = first_output
        # Apply softmax activation
        softmax_output = softmax(first_output)
        return softmax_output

    def back_prop(self, dE_dY, alpha):
        for i, gradient in enumerate(dE_dY):
            if gradient == 0:
                continue
            transformation_eq = np.exp(self.output)
            S_total = np.sum(transformation_eq)

            # Compute gradients with respect to output (Z)
            dY_dZ = -transformation_eq[i]*transformation_eq / (S_total**2)
            dY_dZ[i] = transformation_eq[i]*(S_total - transformation_eq[i]) / (S_total**2)

            # Compute gradients of output Z with respect to weight, bias, input
            dZ_dw = self.flattened_input
            dZ_db = 1
            dZ_dX = self.weight

            # Gradient of loss with respect ot output
            dE_dZ = gradient * dY_dZ

            # Gradient of loss with respect to weight, bias, input
            dE_dw = dZ_dw[np.newaxis].T @ dE_dZ[np.newaxis]
            dE_db = dE_dZ * dZ_db
            dE_dX = dZ_dX @ dE_dZ

            # Update parameters
            self.weight -= alpha*dE_dw
            self.bias -= alpha*dE_db

            return dE_dX.reshape(self.original_shape)

def CNN_forward(image, label, layers):
    output = image/255.
    for layer in layers:
        output = layer.forward_prop(output)
    loss = -np.log(output[label])
    return output, loss

def CNN_backprop(gradient, layers, alpha=0.05):
    grad_back = gradient
    for layer in layers[::-1]:
        if type(layer) in [ConvolutionLayer, SoftmaxLayer]:
            grad_back = layer.back_prop(grad_back,alpha)
        elif type(layer) == MaxPoolingLayer:
            grad_back = layer.back_prop(grad_back)
    return grad_back


def CNN_training(image, label, layers, alpha=0.05):

    # forward step
    output, loss = CNN_forward(image, label, layers)

    # init gradient
    gradient = np.zeros(16)   # replace number with however many classes you have
    gradient[label] = -1/output[label]

    # backprop step
    CNN_backprop(gradient, layers, alpha)

    return output, loss

# function used for testing
def CNN_testing(image,layers):
    pred,loss = CNN_forward(image,0, layers)
    return pred

# get accuracy out of 100
def get_accuracy(preds, y):
    return np.mean(preds == y)*100

# makes a confusion matrix of type numpy array
def confusionMatrix(y_p,y_r,labels):
    confMat = np.zeros((len(labels)+1,len(labels)+1))
    cols = np.array(list(labels.values()))
    confMat[0] = np.append(cols,len(labels))-1
    confMat[:,0] = np.append(cols,len(labels))-1

    for i in range(len(y_p)):
        confMat[int(y_p[i])+1,int(y_r[i])+1] += 1
    return confMat