import tensorflow.compat.v1 as tf
#from tensorflow.python.keras import backend as K
from tensorflow.keras import backend as K
from collections import OrderedDict
from gtc.GTCLayer import GTCLayer, GTCDict, unpackGTCDict


class relu(GTCLayer):
    """
    Implements relu function for both gtc layers
    """
    def __init__(self,
                 quantizer,
                 alpha=0.,
                 max_value=0,
                 threshold=0, name=None, **kwargs):
        self.alpha = alpha
        self.max_value = max_value
        self.threshold = threshold

        super(relu, self).__init__(name=name,
                                   quantizer=quantizer,
                                   trainable=False, **kwargs)

    def call(self, inputs, **kwargs):
        hp_input, lp_input = unpackGTCDict(inputs, error_if_not_gtc_dict=True)

        hp_output = K.relu(hp_input, alpha=self.alpha,  max_value=self.max_value, threshold=self.threshold)
        lp_output = K.relu(lp_input, alpha=self.alpha,  max_value=self.max_value, threshold=self.threshold)

        return GTCDict(hp=hp_output, lp=lp_output)

