
from quantization.quant import Quantization # parent class

from quantization.identity import IdentityQuantization
from quantization.bounded_int import BoundedIntegerQuantization
from quantization.gtc import GTCQuantization
from quantization.linear import LinearQuantization


def GetQuantization(quant_name):
    quant_name = quant_name.lower()
    if quant_name == 'identity':
        return IdentityQuantization
    elif quant_name == 'boundedinteger':
        return BoundedIntegerQuantization
    elif quant_name == 'gtc':
        return GTCQuantization
    elif quant_name == 'linear':
        return LinearQuantization
    else:
        raise ValueError('invalid quantization : %s' % quant_name)



def from_str(quant_type_and_name, kwargs=None):
    if '.' in quant_type_and_name:
        quant_type, quant_name = quant_type_and_name.split('.')
    else:
        quant_type = quant_type_and_name
        quant_name = quant_type

    if kwargs is None:
        kwargs = {}
    quantizer = GetQuantization(quant_type)(name=quant_name, **kwargs)
    return quantizer