# This interface allows you to use layers similar to keras layers.
#  (the names of the GTC layers are aliased to the corresponding layer names in the keras API)
# In your code, include the line 'import gtc'
# Then you can create a layer like this:
#    conv_layer = gtc.Conv2D(...)                # GTC
#    pool_layer = gtc.MaxPooling2D(...)
# similar to creating a keras layer:
#    conv_layer = keras.layers.Conv2D(...)       # Keras
#    pool_layer = keras.layers.MaxPooling2D(...)

from gtc.activation import Activation

from gtc.batch_norm import BatchNormalization
#from gtc.indp_batch_norm import independentBatchNormalization as IndepBatchNorm
#from gtc.shared_batch_norm import sharedBatchNormalization as SharedBatchNorm

from gtc.conv2d import Conv2d as Conv2D
from gtc.conv2d import Conv2d_depthwise as DepthwiseConv2D
from gtc.conv2d_bn import Conv2d_bn as Conv2D_bn
from gtc.conv2d_bn import Conv2d_bn_depthwise as DepthwiseConv2D_bn

from gtc.dense import Dense
from gtc.dense_bn import Dense_bn

from gtc.dropout import Dropout

from gtc.flatten import Flatten, keras_Flatten

from gtc.GTCLayer import GTCLayer
from gtc.gtcModel import GTCModel, GTCDict, unpackGTCDict, singlePrecisionKerasModel
from gtc.gtcModelKerasLayer import GTCModelKerasLayer
from gtc.kerasLayer import GTCKerasLayer

from gtc.lambda_function import Lambda

from gtc.merge import _Merge, Add, Subtract, Multiply, \
    Average, Maximum, Minimum, Concatenate, Dot

from gtc.pooling import Pooling, MaxPooling2D, AveragePooling2D, \
    GlobalMaxPooling2D, GlobalAveragePooling2D

from gtc.preprocess import Preprocess, keras_Preprocess

#from gtc.relu import relu


from gtc.reshape import Reshape
from gtc.resize import Resize, keras_Resize
from gtc.softmax import Softmax

from gtc.squeeze_excite import SqueezeExcite

from gtc.zeropad2d import ZeroPadding2D

from ssd.layers.anchor_boxes import AnchorBoxes, keras_AnchorBoxes
from ssd.layers.l2_normalization import L2Normalization, keras_L2Normalization

# Aliases
AvgPool2D = AveragePooling2D
MaxPool2D = MaxPooling2D
GlobalMaxPool2D = GlobalMaxPooling2D
GlobalAvgPool2D = GlobalAveragePooling2D



def from_str(layer_type_name, keras=False):
    # Remove any artifacts from str()
    #  "<class 'gtc.dense.Dense'>" -> "gtc.dense.Dense"
    #s = s.replace("<class '", '').replace("'>", '')

    # Strip away initial prefixes and keep just the final name
    # "gtc.activation.Activation" ==> "Activation"
    if '.' in layer_type_name:
        layer_type_name = layer_type_name.split('.')[-1]



    all_layers = {
        'Activation': Activation,
        'BatchNormalization': BatchNormalization,
        'Conv2d': Conv2D,
        'Conv2D_bn': Conv2D_bn,
        'DepthwiseConv2D': DepthwiseConv2D,
        'DepthwiseConv2D_bn': DepthwiseConv2D_bn,
        'Dense': Dense,
        'Dense_bn': Dense_bn,
        'Dropout': Dropout,
        'Flatten': Flatten,
        'Lambda': Lambda,

        'Add': Add, 'Subtract': Subtract, 'Multiply': Multiply,
        'Average': Average, 'Maximum': Maximum, 'Minimum': Minimum,
        'Concatenate': Concatenate, 'Dot': Dot,

        'Pooling': Pooling,
        'MaxPooling2D': MaxPooling2D,
        'AveragePooling2D': AveragePooling2D,
        'GlobalMaxPooling2D': GlobalMaxPooling2D,
        'GlobalAveragePooling2D': GlobalAveragePooling2D,

        'Preprocess': Preprocess,
        'Reshape': Reshape,
        'Resize': Resize,
        'Softmax': Softmax,
        'SqueezeExcite': SqueezeExcite,
        'ZeroPadding2D': ZeroPadding2D,

        'AnchorBoxes': AnchorBoxes,
        'L2Normalization': L2Normalization,
    }

    import tensorflow.keras.layers as k

    all_layers_keras = {
        'Activation': k.Activation,
        'BatchNormalization': k.BatchNormalization,
        'Conv2d': k.Conv2D,
        'DepthwiseConv2D': k.DepthwiseConv2D,
        'Dense': k.Dense,
        'Dropout': k.Dropout,
        'Flatten': keras_Flatten,
        'Lambda': k.Lambda,

        'Add': k.Add, 'Subtract': k.Subtract, 'Multiply': k.Multiply,
        'Average': k.Average, 'Maximum': k.Maximum, 'Minimum': k.Minimum,
        'Concatenate': k.Concatenate, 'Dot': k.Dot,

        'MaxPooling2D': k.MaxPooling2D,
        'AveragePooling2D': k.AveragePooling2D,
        'GlobalMaxPooling2D': k.GlobalMaxPooling2D,
        'GlobalAveragePooling2D': k.GlobalAveragePooling2D,

        'Preprocess': keras_Preprocess,
        'Reshape': k.Reshape,
        'Resize': keras_Resize,
        'Softmax': k.Softmax,
        #'SqueezeExcite': k.SqueezeExcite, # Needed for MobilenetV3 (pure-keras version not yet implemented)
        'ZeroPadding2D': k.ZeroPadding2D,

        'AnchorBoxes': keras_AnchorBoxes,
        'L2Normalization': keras_L2Normalization,
    }

    if keras:
        layer = all_layers_keras[layer_type_name]
    else:
        layer = all_layers[layer_type_name]

    return layer
