#Custom L1 distance layer module

#import dependencies
import tensorflow as tf
# from tensorflow.keras.layers import Layer
from keras.layers import Layer  # Use keras directly, not tensorflow.keras


#Siamese L1 Distance class
class L1Dist(Layer):

     #init method - inheritance
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
     #Similarity calculation
    def call(self, input_embedding, validation_embedding):
        # if someone wrapped them in single‑element lists, unwrap here
        if isinstance(input_embedding, (list, tuple)):
            input_embedding = input_embedding[0]
        if isinstance(validation_embedding, (list, tuple)):
            validation_embedding = validation_embedding[0]
        return tf.math.abs(input_embedding - validation_embedding)

    def compute_output_shape(self, input_shape):
        # input_shape is a list/tuple of two shapes (they must match)
        return input_shape[0]
