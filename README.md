# Understanding basics of Neural Network and developing one using Pytorch

## Neural Network Neuron
- Neuron is a basic fundamental unit of a neural network.
- It has summation and an activation function as its component.
- These functions operate on the incoming signals from previous layer neurons or input and gives output after computation.

## Learning rate
- A configurable hyperparameter which implies the rate at which neural network adapts to the given problem.
- It is assumed often as a small positive number,generally in between 0 and 1 and.
- Smaller the learning rate , more time it takes for the network to reach the minimum or converge.(Might even get struck at local minima)
- If the learning rate is large then it causes drastic updates and leads to divergent behavior.
- If the learning rate is just right then it reaches the minimum point swiftly.
- From the equation : 
```W(new) = W(old) - LR(del(error)/del(W(old))```
where W(new) is the new weight and W(old) is the old weight for ith connection. del() is the  partial derivative.
- We can see how LR (Learning Rate) affects the new weight calculation during backpropagation,it controls the rate at which the weight of a signal changes and reaches optimum value to reduce the error.

## Initialization of Weights
- Historically weights in the model are initialized using values drawn randomly from a Gaussian or uniform distribution.
- Recent strategies which are being used for weight initialization:
### Xavier initialization : 
- This approach sets a layer's weight randomly from a uniform distribution bound between ±sqrt(6)/sqrt(ni + ni+1) , where ni is the number of incoming network connections to the layer and ni+1 is the number of outgoing network connections from the layer. 
### Kaiming initialization : 
- This is mainly for neural networks with RELU like activation functions.In this approach we populate weight with numbers randomly chosen from a standard normal distribution. Then multiply these numbers with sqrt(2)/sqrt(n), where n is the number of incoming signals or connections from previous layer to given layer. Then we initialize bias to be zero.
 
## Loss in Neural Network
- Loss in a neural network is the error in prediction. It is the measure in difference between desired value and the obtained value.
- A few examples are Mean Squared Loss, Mean Absolute Loss, e.t.c.

## Chain Rule in Gradient Flow
- It is the partial derivative rule to calculate the derivative of composite function.
- For example, if we have a function Z = h(X,Y) where X = f(t) and Y = g(t) then derivative of Z with respect to t will be calulated as :
```del(Z)/del(t) = (del(Z)/del(X))(del(X)/del(t)) + (del(Z)/del(Y))(del(Y)/del(t))```
