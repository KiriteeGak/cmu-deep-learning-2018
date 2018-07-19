import numpy as np
from math import tanh

data_file_folder="/home/taranjeet/biksha/fwdourhws/handout_hw1p1/demo_test_hw1/data/"
train_file_name="digitstrain.txt"
test_file_name="digitstest.txt"

class Mlp(object):
    def __init__(self,layers,size,activations,iterations=5,learning_rate=5,threshold_error=0.01
                 ,momentum=None,beta=None):
        self.input_size = None
        self.output_size = None
        self.layers = layers
        self.size = size
        self.activations = activations
        self.weights = []
        self.iterations = iterations
        self.bias = []
        self.layer_outputs = []
        self.model = {}
        self.learning_rate = learning_rate
        self.threshold_error = threshold_error
        self.momentum = momentum
        self._velocities = []
        self.beta = beta

    def initialize_weights(self,momentum=False):
        weights=[]
        for _ in range(0,len(self.size)+1):
            if not _:
                if not momentum:
                    weights.append(np.random.rand(self.input_size, self.size[_]) -
                                   np.random.rand(self.input_size, self.size[_]))
                else:
                    self._velocities.append(np.zeros((self.input_size, self.size[_])))
            elif _ == len(self.size):
                if not momentum:
                    weights.append(np.random.rand(self.size[-1], self.output_size) -
                                   np.random.rand(self.size[-1], self.output_size))
                else:
                    self._velocities.append(np.zeros((self.size[-1], self.output_size)))
        return weights

    def _initalise_bias(self,momentum=False):
        if not momentum:
            return [np.ones(_.shape[1]) for _ in self.weights]
        return [np.zeros(_.shape[1]) for _ in self.weights]

    def activation_derivative(self,activation,activation_type=None,derivative=False):
        if activation_type=='relu':
            if not derivative:
                return np.maximum(activation,0)
            return 1.*(activation>0)
        elif activation_type == 'tanh':
            return np.array([tanh(act) if not derivative else None for act in activation[0]])
        elif activation_type == 'sigmoid':
            return np.array([[1 / (1 + np.exp(-1 * act)) if not derivative else None for act in activation[0]]])
        elif activation_type == 'softmax':
            if not derivative:
                sum_ = sum([np.exp(act) for act in activation[0]])
                return np.array([np.exp(act) / sum_ for act in activation])
            return self._helper_gradient_softmax(activation)

    def fit(self,x,y):
        self.input_size = x[0][0].shape[0]
        self.output_size = y[0][0].shape[0]
        self.weights = self.initialize_weights()
        self.bias = self._initialise_bias()
        self.model['weights1'] = self.weights[0]
        self.model['weights2'] = self.weights[1]
        self.model['bias1'] = self.bias[0]
        self.model['bias2'] = self.bias[1]
        self._generate_momentum()
        self.model['momentum_weights1'] = self._velocities[0]
        self.model['momentum_weights2'] = self._velocities[1]
        self.model['momentum_bias1'] = self._velocities[2][0]
        self.model['momentum_bias2'] = self._velocities[2][1]
        iterations_occured = 0
        _error = np.inf
        while iterations_occured < self.iterations and _error > self.threshold_error:
            # X, y = make_datasets(size=1)
            for x, label in zip(X, y):
                self.layer_outputs = self.forward_propagation(x=x)
                weights_bias = self.back_propagation(X=x,
                                                     y=self.layer_outputs[-1],
                                                     output=label,
                                                     a2=self.layer_outputs[1])
                if not self.momentum:
                    self.model['weights1'] -= self.learning_rate * weights_bias[0][0]
                    self.model['bias1'] -= self.learning_rate * weights_bias[0][1]
                    self.model['weights2'] -= self.learning_rate * weights_bias[1][0]
                    self.model['bias2'] -= self.learning_rate * weights_bias[1][1]
                else:
                    self.model['weights1'] -= (self.beta * self.model['momentum_weights1'] +
                                               (1 - self.beta) * weights_bias[0][0]) * self.learning_rate
                    self.model['bias1'] -= (self.beta * self.model['momentum_bias1'] +
                                            (1 - self.beta) * weights_bias[0][1]) * self.learning_rate
                    self.model['weights2'] -= (self.beta * self.model['momentum_weights2'] +
                                               (1 - self.beta) * weights_bias[1][0]) * self.learning_rate
                    self.model['bias2'] -= (self.beta * self.model['momentum_bias2'] +
                                            (1 - self.beta) * weights_bias[1][1]) * self.learning_rate
                    self.model['momentum_weights1'] = self.beta * self.model['momentum_weights1'] + \
                                                            (1 - self.beta) * weights_bias[0][0]
                    self.model['momentum_weights2'] = self.beta * self.model['momentum_weights2'] + \
                                                            (1 - self.beta) * weights_bias[1][0]
                    self.model['momentum_bias1'] = (self.beta * self.model['momentum_bias1'] +
                                                          (1 - self.beta) * weights_bias[0][1])
                    self.model['momentum_bias2'] = (self.beta * self.model['momentum_bias2'] +
                                                          (1 - self.beta) * weights_bias[1][1])

                self.weights = [self.model['weights1'], self.model['weights2']]
                print(self.model['weights2'])
                self.bias = [self.model['bias1'], self.model['bias2']]

                iterations_occured += 1
                _error = self.error_(output=self.layer_outputs[-1], actual=label, type_="crossentropy")
                print("Classification error at the current time is {}".format(round(_error, 3)))

    @staticmethod
    def _helper_gradient_softmax(array_):
        grads = []
        den_ = sum([np.exp(e) for e in array_]) ** 2
        for i, _ in array_:
            _temp_array = array_.copy()
            _temp_array.pop(i)
            grads.append(np.exp(array_[i]) * (sum([np.exp(e) for e in _temp_array])) / den_)
        return grads

    def _generate_momentum(self):
        self.initialize_weights(momentum=True)
        self._velocities.append(self._initialise_bias(momentum=True))

    def _initialise_bias(self, momentum=False):
        if not momentum:
            return [np.ones(_.shape[1]) for _ in self.weights]
        return [np.zeros(_.shape[1]) for _ in self.weights]


    def forward_propagation(self, x):
        layer_outputs = []
        for i, (w, activation) in enumerate(zip(self.weights, self.activations)):
            a = np.array(x).dot(w) + self.bias[i]
            z = self.activation_derivative(a, activation_type=activation)
            layer_outputs.append(a)
            layer_outputs.append(z)
            x = z
        return layer_outputs

    def back_propagation(self, X, y, a2=None, output=None, reg_lambda=0.01):
        delta2 = self.error_(output=output, actual=y, type_="crossentropy", derivative=True)
        dW2 = a2.T.dot(delta2)
        db2 = np.sum(delta2, axis=0)
        delta1 = delta2.dot(self.weights[1].T) * self.activation_derivative(a2,
                                                                             activation_type="relu",
                                                                             derivative=True)
        dW1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0)
        return (dW1, db1), (dW2, db2)

    @classmethod
    def error_(cls, output, actual, type_=None, epsilon=1e-12, derivative=False):
        if type_ == "crossentropy":
            if not derivative:
                predictions = np.clip(output, epsilon, 1. - epsilon)
                n = predictions.shape[0]
                ce = -np.sum(np.sum(actual * np.log(predictions + 1e-9))) / n
                return ce
            return np.array([cls._helper_derivative_softmax(act, out) for (act, out) in zip(actual, output)])
        else:
            raise NotImplementedError("not implemented() for square")

    @staticmethod
    def _helper_derivative_softmax(actual, output):
        return np.array([-1 * ((act * 1 / (out if out else np.inf)) +
                               (1 - act) * (1 / (1 - (out if out != 1 else np.inf)))) for (act, out) in
                         zip(actual, output)])
def read_data(file_path):
    with open(file_path) as data_file:
        vector,label=[],[]
        for r in data_file.readlines():
            r=r.split(",")
            data=[float(i) for i in r]
            vector.append(np.array([data[:-1]]))
            label.append(data[-1])
    #converting labels into an vector data
    y=[]
    for i in label:
        label_vector=list(np.zeros(10,dtype="int32"))
        label_vector[int(i)]=1
        y.append(np.array([label_vector]))
    return vector,y




if __name__== "__main__":
    X,y=read_data(data_file_folder+train_file_name)
    s = Mlp(layers=1,
                             size=[2],
                             activations=('relu', 'softmax'),
                             iterations=2000,
                             learning_rate=0.01,
                             threshold_error=0.15,
                             momentum=False,
                             beta=0.9)
    s.fit(x=X, y=y)

