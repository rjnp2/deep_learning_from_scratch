import cupy as cp
from . import Layer
from deeplearning.activat import  ReLU

class Activation(Layer):

    """
    A layer that applies an activation operation to the input.
    
    Parameters
    ----------
    activation : str
        ctivation function, such as tf.nn.relu, or string name of
        built-in activation function, such as "relu".
        
    """

    def __init__(self, activation : str):

        self.trainable = True      
        self.activation_func = activation()
 
    def layer_name(self):
        '''
                
        Returns
        -------
        str
            Name of activation function.

        '''

        return "Activation (%s)" % (self.activation_func.__class__.__name__)

    def forward_pass(self, X: cp.array,
                     training: str =True) -> cp.array:
        '''
        
        Parameters
        ----------
        X : cp.array
            array of prevoius layer of deep-learning.
        training : str, optional
            training or not. The default is True.

        Returns
        -------
        cp.array
            array after activation appplies.

        '''
        
        if training: 
            self.layer_input = (X.get()).copy()

        return self.activation_func(X)

    def backward_pass(self, accum_grad: cp.array ) -> cp.array:
        '''
        
        Parameters
        ----------
        accum_grad : cp.array
            gradient with respect to weight.

        Returns
        -------
        cp.array
            gradient of activation.

        '''
        accum_grad = accum_grad * self.activation_func.gradient(cp.asarray(self.layer_input))
        cp.cuda.Stream.null.synchronize()

        del self.layer_input
        
        return accum_grad

    def determin_output_shape(self):
        '''
        Return input shape of this object layers
        '''
        return self.input_shape
