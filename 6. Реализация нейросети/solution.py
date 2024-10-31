from interface import *


# ================================= 1.4.1 SGD ================================
class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter

            :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam

                :return: np.array, new parameter values
            """
            # your code here \/
            return parameter - self.lr * parameter_grad
            # your code here /\

        return updater


# ============================= 1.4.2 SGDMomentum ============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self.lr = lr
        self.momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter

            :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam

                :return: np.array, new parameter values
            """
            # your code here \/
            updater.inertia = self.momentum * updater.inertia + self.lr * parameter_grad
            return parameter - updater.inertia
            # your code here /\

        updater.inertia = np.zeros(parameter_shape)
        return updater


# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, ...)), input values

            :return: np.array((n, ...)), output values

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        return np.maximum(inputs, 0)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

            :return: np.array((n, ...)), dLoss/dInputs

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        mask = self.forward_inputs >= 0
        return mask * grad_outputs
        # your code here /\


# =============================== 2.1.2 Softmax ==============================
class Softmax(Layer):
    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, d)), output values

                n - batch size
                d - number of units
        """
        # your code here \/
        exp = np.exp(inputs - inputs.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of units
        """
        # your code here \/
        softmax = self.forward_outputs
        diag = softmax[:, :, None] * (1 - softmax[:, None, :])
        off_diag = -softmax[:, :, None] * softmax[:, None, :]
        jacobian = np.where(np.eye(grad_outputs.shape[1], dtype=bool), diag, off_diag)
        return (grad_outputs[:, None, :] @ (jacobian))[:, 0, :]
        # your code here /\


# ================================ 2.1.3 Dense ===============================
class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_units = units

        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_units, = self.input_shape
        output_units = self.output_units

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name='weights',
            shape=(input_units, output_units),
            initializer=he_initializer(input_units)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_units,),
            initializer=np.zeros
        )

        self.output_shape = (output_units,)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, c)), output values

                n - batch size
                d - number of input units
                c - number of output units
        """
        # your code here \/
        return inputs @ self.weights + self.biases
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of input units
                c - number of output units
        """
        # your code here \/
        self.weights_grad = self.forward_inputs.T @ grad_outputs
        self.biases_grad = np.ones(grad_outputs.shape[0]) @ grad_outputs
        return grad_outputs @ self.weights.T
        # your code here /\


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def value_impl(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values

            :return: np.array((1,)), mean Loss scalar for batch

                n - batch size
                d - number of units
        """
        # your code here \/
        return -np.sum(np.log(np.sum(y_pred * y_gt, axis=1))).reshape(1,) / y_gt.shape[0]
        # your code here /\

    def gradient_impl(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values

            :return: np.array((n, d)), dLoss/dY_pred

                n - batch size
                d - number of units
        """
        # your code here \/
        y = y_pred.copy()
        y[y <= eps] = eps
        return - y_gt / y / y_gt.shape[0]
        # your code here /\


# ======================== 2.3 Train and Test on MNIST =======================
def train_mnist_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    model = Model(loss=CategoricalCrossentropy(), optimizer=SGDMomentum(lr=0.001, momentum=0.2))

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Dense(units=256, input_shape=(784,)))
    model.add(ReLU())
    model.add(Dense(units=64))
    model.add(ReLU())
    model.add(Dense(units=10))
    model.add(Softmax())

    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(x_train, y_train, batch_size=16, epochs=5, x_valid=x_valid, y_valid=y_valid)

    # your code here /\
    return model


# ============================== 3.3.2 convolve ==============================
def convolve(inputs, kernels, padding=0):
    """
        :param inputs: np.array((n, d, ih, iw)), input values
        :param kernels: np.array((c, d, kh, kw)), convolution kernels
        :param padding: int >= 0, the size of padding, 0 means 'valid'

        :return: np.array((n, c, oh, ow)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
    """
    # !!! Don't change this function, it's here for your reference only !!!
    assert isinstance(padding, int) and padding >= 0
    assert inputs.ndim == 4 and kernels.ndim == 4
    assert inputs.shape[1] == kernels.shape[1]

    if os.environ.get('USE_FAST_CONVOLVE', False):
        return convolve_pytorch(inputs, kernels, padding)
    else:
        return convolve_numpy(inputs, kernels, padding)


def convolve_numpy(inputs, kernels, padding):
    """
        :param inputs: np.array((n, d, ih, iw)), input values
        :param kernels: np.array((c, d, kh, kw)), convolution kernels
        :param padding: int >= 0, the size of padding, 0 means 'valid'

        :return: np.array((n, c, oh, ow)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
    """
    # your code here \/
    n, d, ih, iw = inputs.shape
    c, _, kh, kw = kernels.shape
    oh, ow = ih - kh + 1 + 2* padding, iw - kw + 1 + 2* padding
    kernels = kernels[:, :, ::-1, ::-1]
    
    if padding > 0:
        inputs = np.pad(inputs, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    
    outputs = np.zeros(n*c*oh*ow).reshape(n,c,oh,ow)
    for i in range(oh):
        for j in range(ow):
            outputs[:,:,i,j] = np.einsum('ndhw,cdhw->nc', inputs[:,:,i:i+kh,j:j+kw], kernels)
    return outputs
    # your code here /\


# =============================== 4.1.1 Conv2D ===============================
class Conv2D(Layer):
    def __init__(self, output_channels, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert kernel_size % 2, "Kernel size should be odd"

        self.output_channels = output_channels
        self.kernel_size = kernel_size

        self.kernels, self.kernels_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        output_channels = self.output_channels
        kernel_size = self.kernel_size

        self.kernels, self.kernels_grad = self.add_parameter(
            name='kernels',
            shape=(output_channels, input_channels, kernel_size, kernel_size),
            initializer=he_initializer(input_h * input_w * input_channels)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_channels,),
            initializer=np.zeros
        )

        self.output_shape = (output_channels,) + self.input_shape[1:]

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, c, h, w)), output values

                n - batch size
                d - number of input channels
                c - number of output channels
                (h, w) - image shape
        """
        # your code here \/
        return convolve(inputs, self.kernels, self.kernel_size // 2) + self.biases[None,:,None,None]
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c, h, w)), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of input channels
                c - number of output channels
                (h, w) - image shape
        """
        # your code here \/
        self.biases_grad = np.einsum('nchw->c', grad_outputs)
        
        self.kernels_grad = np.transpose(convolve(np.transpose(self.forward_inputs[:,:,::-1,::-1], (1,0,2,3)), np.transpose(grad_outputs, (1,0,2,3)), self.kernel_size // 2),  (1,0,2,3))
        inv_pad = self.kernel_size - self.kernel_size // 2 - 1
        grad_inputs = convolve(grad_outputs, np.transpose(self.kernels[:,:,::-1,::-1], (1,0,2,3)), inv_pad)
        return grad_inputs
        # your code here /\


# ============================== 4.1.2 Pooling2D =============================
class Pooling2D(Layer):
    def __init__(self, pool_size=2, pool_mode='max', *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert pool_mode in {'avg', 'max'}

        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.forward_idxs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        channels, input_h, input_w = self.input_shape
        output_h, rem_h = divmod(input_h, self.pool_size)
        output_w, rem_w = divmod(input_w, self.pool_size)
        assert not rem_h, "Input height should be divisible by the pool size"
        assert not rem_w, "Input width should be divisible by the pool size"

        self.output_shape = (channels, output_h, output_w)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, ih, iw)), input values

            :return: np.array((n, d, oh, ow)), output values

                n - batch size
                d - number of channels
                (ih, iw) - input image shape
                (oh, ow) - output image shape
        """
        # your code here \/
        n, d, ih, iw = inputs.shape
        #d, output_h, output_w = self.output_shape
        p = self.pool_size
        unfolded = np.lib.stride_tricks.sliding_window_view(inputs, (p, p), axis=(-2,-1))[:,:,::p,::p,...]
        
        if self.pool_mode == 'max':
            outputs = unfolded.max(axis=(-2,-1))
            
            flatten = np.reshape(unfolded, (n,d,-1,p*p))
            argmax = np.argmax(flatten, axis=-1)
            mask = np.zeros_like(flatten)
            np.put_along_axis(mask, argmax[...,None], 1, axis=-1)
            self.forward_idxs = mask.reshape(unfolded.shape).swapaxes(3, 4).reshape(inputs.shape)
        else:
            outputs = unfolded.mean(axis=(-2,-1))
            
            
        return outputs
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d, oh, ow)), dLoss/dOutputs

            :return: np.array((n, d, ih, iw)), dLoss/dInputs

                n - batch size
                d - number of channels
                (ih, iw) - input image shape
                (oh, ow) - output image shape
        """
        # your code here \/
        p = self.pool_size
        if self.pool_mode == 'max':
            grad_inputs = np.kron(grad_outputs, np.ones((p,p))) * self.forward_idxs
        else:
            grad_inputs = np.kron(grad_outputs, np.ones((p,p)))/p**2
        return grad_inputs
        # your code here /\


# ============================== 4.1.3 BatchNorm =============================
class BatchNorm(Layer):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

        self.running_mean = None
        self.running_var = None

        self.beta, self.beta_grad = None, None
        self.gamma, self.gamma_grad = None, None

        self.forward_inverse_std = None
        self.forward_centered_inputs = None
        self.forward_normalized_inputs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        self.running_mean = np.zeros((input_channels,))
        self.running_var = np.ones((input_channels,))

        self.beta, self.beta_grad = self.add_parameter(
            name='beta',
            shape=(input_channels,),
            initializer=np.zeros
        )

        self.gamma, self.gamma_grad = self.add_parameter(
            name='gamma',
            shape=(input_channels,),
            initializer=np.ones
        )

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, d, h, w)), output values

                n - batch size
                d - number of channels
                (h, w) - image shape
        """
        # your code here \/
        if self.is_training:
            mu = inputs.mean(axis=(0,2,3))
            self.forward_centered_inputs = inputs - mu[None, :, None, None] #ndhw
            nu = inputs.var(axis=(0,2,3))
            self.forward_inverse_std = 1 / np.sqrt(eps+nu[None, :, None, None]) #ndhw
            self.forward_normalized_inputs = self.forward_centered_inputs * self.forward_inverse_std #ndhw
            self.running_mean = self.momentum * self.running_mean - (1 - self.momentum) * mu #d
            self.running_var = self.momentum * self.running_var - (1 - self.momentum) * nu #d
        else:
            # self.forward_centered_inputs = inputs - self.running_mean[None, :, None, None]
            # self.forward_inverse_std = 1 / np.sqrt(eps+self.running_var[None, :, None, None])
            # self.forward_normalized_inputs = self.forward_centered_inputs * self.forward_inverse_std
            self.forward_normalized_inputs = (inputs - self.running_mean[None, :, None, None]) / np.sqrt(eps+self.running_var[None, :, None, None])
        
        outputs = self.gamma[None, :, None, None] * self.forward_normalized_inputs + self.beta[None, :, None, None] #ndhw
        return outputs
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d, h, w)), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs
 
                n - batch size
                d - number of channels
                (h, w) - image shape
        """
        # your code here \/
        n, d, h, w = grad_outputs.shape
        self.beta_grad = np.einsum('ndhw->d', grad_outputs)
        self.gamma_grad = np.einsum('ndhw,ndhw->d', grad_outputs, self.forward_normalized_inputs)
        
        grad_normalizedx = grad_outputs * self.gamma[None, :, None, None]
        grad_lnu =  -0.5 * np.einsum('ndhw,ndhw->d', grad_normalizedx, self.forward_centered_inputs)[None,:,None,None] * (self.forward_inverse_std**3)
        grad_lmu = -np.einsum('ndhw,ndhw->d', grad_normalizedx,  self.forward_inverse_std)[None,:,None,None] -2*grad_lnu * np.einsum('ndhw->d', self.forward_centered_inputs)[None,:,None,None]/(n*h*w)
        grad_inputs = grad_normalizedx * self.forward_inverse_std + 2 * grad_lnu * self.forward_centered_inputs/(n*h*w) + grad_lmu/(n*h*w)
        return grad_inputs
        # your code here /\



# =============================== 4.1.4 Flatten ==============================
class Flatten(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = (np.prod(self.input_shape),)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, (d * h * w))), output values

                n - batch size
                d - number of input channels
                (h, w) - image shape
        """
        # your code here \/
        return inputs.reshape(inputs.shape[0], self.output_shape[0])
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, (d * h * w))), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of units
                (h, w) - input image shape
        """
        # your code here \/
        d,h,w = self.input_shape
        return grad_outputs.reshape(grad_outputs.shape[0], d, h, w)
        # your code here /\


# =============================== 4.1.5 Dropout ==============================
class Dropout(Layer):
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.forward_mask = None

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, ...)), input values

            :return: np.array((n, ...)), output values

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        if self.is_training:
            self.forward_mask = np.random.choice([0,1], size=inputs.shape, p=[self.p, 1 - self.p])
            outputs = inputs * self.forward_mask
        else:
            outputs = (1-self.p) * inputs
        return outputs
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

            :return: np.array((n, ...)), dLoss/dInputs

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        return grad_outputs*self.forward_mask
        # your code here /\


# ====================== 2.3 Train and Test on CIFAR-10 ======================
def train_cifar10_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    #1) Create a Model
    model = Model(CategoricalCrossentropy(), SGDMomentum(lr = 0.1, momentum = 0.9))

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Conv2D(output_channels=16, kernel_size=3, input_shape=(3,32,32)))
    model.add(ReLU())
    model.add(Pooling2D(pool_mode='avg'))
    model.add(Conv2D(output_channels=64,kernel_size=3))
    model.add(ReLU())
    model.add(Pooling2D(pool_mode='avg'))
    model.add(Conv2D(output_channels=32,kernel_size=3))
    model.add(ReLU())
    model.add(Pooling2D(pool_mode='avg'))
    model.add(Flatten())
    #model.add(Dropout(p=0.2))
    model.add(Dense(units=10))
    model.add(Softmax())

    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(x_train, y_train, batch_size=256, epochs=11, x_valid=x_valid, y_valid=y_valid)

    # your code here /\
    return model

# ============================================================================
