import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        return 1 if nn.as_scalar(self.run(x)) >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        has_incorrect = True
        while has_incorrect:
            has_incorrect = False
            for x, y in dataset.iterate_once(1):
                run = self.run(x)
                pred = self.get_prediction(x)
                scalar_y = nn.as_scalar(y)
                if scalar_y != pred:
                    has_incorrect = True
                    self.w.update(x, scalar_y)

class DenseLayer:
    """
    A generic densely connected layer.
    units: int
        number of neurons
    prev_units: int
        number of neurons of the previous layer.
    dimensions: tuple
        the shape of the weights in this dense layer.
    activation: function
        for each neuron, applies the activation function on the neuron, if None, it is the identity function
    bias: bool
        whether or not this dense layer adds a common bias weight to every neuron
    """
    def __init__(self, units, prev_units, dimensions=(1,1), activation=nn.ReLU, bias=True):
        self.params = [nn.Parameter(*dimensions) for _ in range(prev_units*units)]
        self.bias = bias
        self.prev_units = prev_units
        self.units = units
        if self.bias:
            self.bias_w = nn.Parameter(*dimensions)
        self.activation = activation
        if activation is None:
            self.activation = lambda x: x

    def get_result(self, other):
        """
        Other is a list of results
        Returns a list of results that are len(units).
        """
        results = []
        for i in range(self.units):
            so_far = None
            for j, o in enumerate(other):
                if so_far is None:
                    so_far = nn.Linear(o, self.params[i*self.prev_units + j])
                else:
                    so_far = nn.Add(nn.Linear(o, self.params[i*self.prev_units + j]), so_far)
            if self.bias:
                so_far = nn.AddBias(so_far, self.bias_w)
            results.append(self.activation(so_far))
        return results

    def get_params(self):
        if self.bias:
            return self.params + [self.bias_w]
        else:
            return self.params

    def __repr__(self):
        return f"DenseLayer with {self.units} neuron(s)"


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        self.lr = 0.0005
        self.batch_size = 1
        self.num_layers = 4
        self.neurons = 5
        # Initial input layer
        self.layers = [DenseLayer(units=self.neurons, prev_units=1, dimensions=(1,1), activation=nn.ReLU, bias=True)]
        self.layers.extend([DenseLayer(units=self.neurons, prev_units=self.neurons, dimensions=(1,1), activation=nn.ReLU, bias=True) for _ in range(1, self.num_layers - 1)])
        # Add final output layer
        self.layers.append(DenseLayer(units=1, prev_units=self.neurons, dimensions=(1,1), activation=None, bias=False))
        print("LAYERS:", self.layers)
        self.params = []
        for layer in self.layers:
            self.params.extend(layer.get_params())

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        prev_layer_result = self.layers[0].get_result([x])
        for i in range(1, len(self.layers)):
            prev_layer_result = self.layers[i].get_result(prev_layer_result)
        return prev_layer_result[0]

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for trainings
        Returns: a loss node
        """
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        losses = [1.0]
        i = 1
        while not sum(losses) / len(losses) <= 0.02:
            print(f"Epoch {i}")
            losses = []
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, self.params)
                for j, param in enumerate(self.params):
                    param.update(gradients[j], -self.lr)
            # Validate
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                losses.append(nn.as_scalar(loss))
            i += 1

"""
                 !#########       #
               !########!          ##!
            !########!               ###
         !##########                  ####
       ######### #####                ######
        !###!      !####!              ######
          !           #####            ######!
                        !####!         #######
                           #####       #######
                             !####!   #######!
                                ####!########
             ##                   ##########
           ,######!          !#############
         ,#### ########################!####!
       ,####'     ##################!'    #####
     ,####'            #######              !####!
    ####'
"""

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.lr = 0.008
        self.batch_size = 50
        self.hidden_size = 200
        self.w0 = nn.Parameter(784, self.hidden_size)
        self.w1 = nn.Parameter(self.hidden_size, 10)
        self.b0 = nn.Parameter(1, self.hidden_size)
        self.b1 = nn.Parameter(1, 10)
        # self.w2 = nn.Parameter(self.hidden_size, 10)
        # self.b2 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        r1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w0), self.b0))
        return nn.AddBias(nn.Linear(r1, self.w1), self.b1)
        # r2 = nn.ReLU(nn.AddBias(nn.Linear(r1, self.w1), self.b1))
        # return nn.AddBias(nn.Linear(r2, self.w2), self.b2)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        accuracy = 0
        while accuracy < 0.975:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                # gradients = nn.gradients(loss, [self.w0, self.w1, self.w2, self.b0, self.b1, self.b2])
                gradients = nn.gradients(loss, [self.w0, self.w1, self.b0, self.b1])
                self.w0.update(gradients[0], -self.lr)
                self.w1.update(gradients[1], -self.lr)
                # self.w2.update(gradients[2], -self.lr)
                self.b0.update(gradients[2], -self.lr)
                self.b1.update(gradients[3], -self.lr)
                # self.b2.update(gradients[5], -self.lr)

            accuracy = dataset.get_validation_accuracy()

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.lr = 0.08
        self.batch_size = 65
        self.hidden_size = 300
        self.w0 = nn.Parameter(self.num_chars, self.hidden_size)
        self.b0 = nn.Parameter(1, self.hidden_size)
        self.w1 = nn.Parameter(self.hidden_size, self.hidden_size)
        self.b1 = nn.Parameter(1, self.hidden_size)
        self.w2 = nn.Parameter(self.hidden_size, 5)
        # self.b2 = nn.Parameter(1, 5)


    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        z = nn.Linear(xs[0], self.w0)
        h = nn.AddBias(nn.Linear(nn.ReLU(nn.AddBias(z, self.b0)), self.w1), self.b1)
        for i in range(1, len(xs)):
            z = nn.Add(nn.Linear(xs[i], self.w0), nn.Linear(h, self.w1))
            h = nn.AddBias(nn.ReLU(z), self.b0)
            i += 1
        return nn.Linear(h, self.w2)


    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        accuracy = 0
        while accuracy < 0.815:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                # gradients = nn.gradients(loss, [self.w0, self.w1, self.w2, self.b0, self.b1, self.b2])
                gradients = nn.gradients(loss, [self.w0, self.w1, self.b0, self.b1])
                self.w0.update(gradients[0], -self.lr)
                self.w1.update(gradients[1], -self.lr)
                # self.w2.update(gradients[2], -self.lr)
                self.b0.update(gradients[2], -self.lr)
                self.b1.update(gradients[3], -self.lr)
                # self.b2.update(gradients[5], -self.lr)

            accuracy = dataset.get_validation_accuracy()
