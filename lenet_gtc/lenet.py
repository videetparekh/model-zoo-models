import leip.compress.training.gtc

# from mnist.config import config
from leip.compress.training import gtc, gtcModel
from leip.compress.training.quantization import IdentityQuantization
from leip.compress.training.quantization import GTCQuantization
from leip.compress.training.utility_scripts import quant_utils as quant
from leip.compress.training.networks import net_utils as net

import leip.compress.training.gtcModel


# create LeNet model using keras Sequential
def LeNet(args, x_placeholder):
    quantizers = quant.QuantizerSet(num_layers=5)

    L1Model = gtcModel.GTCModel()
    L1Model.add(x_placeholder)

    net.add_conv_bn_act_layers(L1Model, filters=5, kernel_size=5, use_bias=False, args=args, quantizer=quantizers)
    L1Model.add(gtc.MaxPooling2D(strides=2, pool_size=2))

    net.add_conv_bn_act_layers(L1Model, filters=10, kernel_size=5, use_bias=False, args=args, quantizer=quantizers)
    L1Model.add(gtc.MaxPooling2D(strides=2, pool_size=2))

    L1Model.add(gtc.Flatten())
    L1Model.add(gtc.Dense(quantizer=quantizers.next(), units=128, use_bias=False))
    L1Model.add(gtc.Activation(quantizer=GTCQuantization(), activation='relu'))

    L1Model.add(gtc.Dense(quantizer=quantizers.next(), units=128, use_bias=False))
    L1Model.add(gtc.Activation(quantizer=IdentityQuantization(), activation='relu'))

    L1Model.add(gtc.Dense(quantizer=quantizers.next(), units=args.num_classes, use_bias=False))

    quantizers.confirm_done()
    return L1Model
