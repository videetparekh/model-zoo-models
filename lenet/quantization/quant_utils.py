import numpy as np
import quantization as quant
from utility_scripts import misc_utl as utl

class QuantizerSet():
    def __init__(self, num_layers=None, num_quant=None, pre_skip=0, post_skip=0,
                 layer_quant_idxs=None,
                 all_identity=False,
                 idxs_identity=None, idxs_quant=None,
                 conv_quant='gtc', bn_quant='identity', act_quant='identity',
                 num_bits_linear=None, linear_bits_fixed=False,
                 prefix='', config=None, idx_start=0):

        """ Class to handle allocation of quantizers to layers in a network.
        :param num_layers: the number of layers in the network.
        :param num_quant: the number of unique quantizers.
            If num_quant=1, all the layers will receive the same quantizer.
            If num_quant=num_layers, each layer will receive its own quantizer.
            If 1 < num_quant < num_layers, the specified number of quantizers will be
            shared evenly across the number of layers.
        :param pre_skip/post_skip: when specifying num_quant to share the quantizers,
            you may want to skip the first few (pre_skip) or last few layers (post_skip).

        :param layer_quant_idxs: Instead of specifying num_quant, if you want,
        you can manually specify which layers receive which quantizers by providing
        a list of indices. e.g.:
            [1,1,1,2,2,2,3,3,3]
        for a network of 9 layers: will allocate quantizer #1 to the first 3 layers,
        quantizer #2 for the second three layers, and quantizer #3 to the last three layers
        (this is what will happen automatically if you specify num_layers=9, num_quant=3)
        Use `None` for a particular layer to indicate no quantization(ie. identity)

        :param all_identity: flag to make make all the layers use IdentityQuantization
        :param idxs_identity: a list of integer indices, if provided:
            make only the quantizers for these layers use IdentityQuantization
            can have negative indices to index from the end: eg.
            [-2,-1] = second last and last layers
        :param idxs_quant: (alternative to providing idxs_identity)
            make only the quantizers for these layers NOT use IdentityQuantization

        :param conv_quant: the quantizer used for convolutional and dense layers
            (can be 'gtc', 'linear', 'identity', or any valid quantizer name.
            (see the GetQuantization function in the __init__.py API)
        :param bn_quant: quantizer provided to the batch-normalization layer
        :param act_quant: quantizer provided to the activation layer
        :param config: a NetConfig class that can hold many of the above parameters,
        for ease in passing around as argument.

        Usage: When building a network, use the quantizer to allocate quantizers to
        successive layers.  Because networks are often built with a conv-batchnorm-activation
        motif, this class returns dict objects with 'conv', 'batchnorm' and 'activation' keys
        Use the 'next()' method to retrieve this set of quantizers.
        e.g.:

            quant_set = QuantizerSet(num_layers=9, num_quant=3,
                                     conv_quant='gtc', bn_quant='linear', act_quant='identity')

            quant_dict1 = quant_set.next()
            model.add( gtc.Conv2D( filters=32, quantizer=quant_dict1['conv'] ) )
            model.add( gtc.BatchNormalization( quantizer=quant_dict1['batchnorm'] ) )
            model.add( gtc.Activation('relu', quantizer=quant_dict1['activation'] ) )

            quant_dict2 = quant_set.next()
            model.add( gtc.Conv2D( filters=64, quantizer=quant_dict2['conv'] ) )
            model.add( gtc.BatchNormalization( quantizer=quant_dict2['batchnorm'] ) )
            model.add( gtc.Activation('relu', quantizer=quant_dict2['activation'] ) )

        For convenience, you can use the conv_bn_act_layers() function in net_utils.py
        which has been designed to accept the full quantizer dictionary and allocate
        it to the conv, batchnorm, and activation layers respectively

            add_conv_bn_act_layers(model, filters=32, activation='relu',
                                    quant_dict=quant_set.next())
            add_conv_bn_act_layers(model, filters=64, activation='relu',
                                    quant_dict=quant_set.next())
        """

        if config is not None:
            conv_quant = config.conv_quant
            bn_quant = config.bn_quant
            act_quant = config.act_quant

            all_identity= config.identity_quantization
            idxs_quant = config.idxs_quant
            num_quant = config.num_quantizers
            pre_skip = config.quant_pre_skip
            post_skip = config.quant_post_skip
            num_bits_linear = config.num_bits_linear
            linear_bits_fixed = config.linear_bits_fixed

        if all_identity:
            conv_quant, bn_quant, act_quant = 'identity', 'identity', 'identity'


        if num_layers is not None:
            if num_quant is None:
                num_quant = num_layers  # default = all quantizers distinct
                #num_quant = 1            # default = all quantizers the same

            if isinstance(num_quant, str) and num_quant == 'indep':
                num_quant = num_layers - pre_skip - post_skip

            layer_quant_idxs = getEvenlySpacedQuantIdxs(
                num_layers, num_quant, pre_skip, post_skip)

        self._quant_idxs = layer_quant_idxs

        # create a list of unique quantizers
        unique_quantizers = []

        # pad names with appropriate number of zeros so sorting keeps order:  2 --> 02
        num_width = int(np.ceil( np.log10(num_quant) ))

        if idxs_identity is not None:
            # idxs_identity can have negative numbers, referring to last layers
            idxs_identity = [x if x >= 0 else x + num_layers for x in idxs_identity]
        if idxs_quant is not None:
            idxs_quant = [x if x >= 0 else x + num_layers for x in idxs_quant]


        def init_quant(quant_type, q_id, suffix=''):
            name = '%s%0*d%s' % (quant_type[0].upper(), num_width, q_id + 1, suffix)
            init_kwargs = {'name': name}
            if prefix:
                init_kwargs['variable_scope'] = prefix

            if quant_type == 'linear':
                if num_bits_linear is not None:
                    init_kwargs['num_bits'] = num_bits_linear
                if linear_bits_fixed is not None:
                    init_kwargs['train'] = not bool(linear_bits_fixed)


            return quant.GetQuantization(quant_type)(**init_kwargs)

        for q_idx in range(num_layers):
            use_identity = None


            if idxs_identity is not None:
                use_identity = q_idx in idxs_identity
            elif idxs_quant is not None:
                use_identity = q_idx not in idxs_quant

            if all_identity: # this overrides all other considerations
                use_identity = True

            if use_identity:
                conv_quant_i, bn_quant_i, act_quant_i = 'identity', 'identity', 'identity'
            else:
                conv_quant_i, bn_quant_i, act_quant_i = conv_quant, bn_quant, act_quant

            suffixes = dict(conv='', bn='', act='')
            # suffixes for disambiguation in case two (or all three) use the same quantizer
            # We look at all three pairs of quantizers, and add a letter to
            # disambiguate if they are the same.
            if conv_quant_i == bn_quant_i and conv_quant_i != 'identity':
                suffixes['conv'] = 'c'
                suffixes['bn'] = 'b'
            if conv_quant_i == act_quant_i and conv_quant_i != 'identity':
                suffixes['conv'] = 'c'
                suffixes['act'] = 'a'
            if bn_quant_i == act_quant_i and bn_quant_i != 'identity':
                suffixes['bn'] = 'b'
                suffixes['act'] = 'a'

            q_idx_offset = q_idx + idx_start
            unique_quantizers.append(
                {'conv': init_quant(conv_quant_i, q_idx_offset, suffix=suffixes['conv']),
                 'batchnorm': init_quant(bn_quant_i, q_idx_offset, suffix=suffixes['bn']),
                 'activation': init_quant(act_quant_i, q_idx_offset, suffix=suffixes['act']) }
            )

        # using the idxs, assemble the final list of (possibly duplicated) quantizers.
        quantizers_use = []
        skip_quant = {k: init_quant('identity', -1, '_') for k in unique_quantizers[0].keys()}

        for q_idx in range(num_layers):
            quant_idx = layer_quant_idxs[q_idx]
            if quant_idx is None:
                quantizers_use.append(skip_quant)
            else:
                quantizers_use.append(unique_quantizers[layer_quant_idxs[q_idx] - 1])

        self._quantizers = quantizers_use
        self._idx = 0

    def __getitem__(self, item):
        return self._quantizers[item]

    def print_pattern(self):
        set_name = lambda s:  '[' + s['conv'].name[1:] + ':' + s['conv'].name[0] + s['batchnorm'].name[0] + s['activation'].name[0] + ']'

        all_names = [ set_name(s) for s in self._quantizers]
        all_quant_names = utl.concat_detect_repeats(all_names)
        print('Quantizers : %s ' % all_quant_names)

    def next(self, layer=None, idx=None):

        if self._idx >= len(self._quantizers):
            raise ValueError('Invalid idx %d: only have %d quantizers' % (self._idx, len(self._quantizers)))

        quant_dict = self._quantizers[self._idx]

        if idx is None:
            self._idx += 1

        if layer is None:
            return quant_dict
        else:
            return quant_dict[layer]

    def confirm_done(self, print_only=False):
        if print_only:
            print('Note: Used %d out of %d quantizers' % (self._idx, len(self._quantizers)))

        elif self._idx != len(self._quantizers):
            raise ValueError('Only used %d out of %d quantizers' % (self._idx, len(self._quantizers)))


def get_quant_idxs(block_reps, offset=1, mult=1):
    quant_idxs = []
    idx = offset
    for block_rep in block_reps:
        n = block_rep * mult
        #quant_idxs.append( list(range(idx, idx+n))  )
        quant_idxs.append( slice(idx, idx + n) )
        idx += n
    return quant_idxs


def getFixedResnetQuantIdxs(num_layers=23, num_quant=None, identity=False):
    if num_quant is None:
        num_quant = num_layers

    if num_quant == 23:
        quant_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] # all quantizers unique

    elif num_quant == 12:
        quant_idxs = [1, 1,  2, 2,  3, 3,  4, 4,  5, 5,  6, 6,  7, 7,  8, 8,  9, 9,  10, 10,  11, 11, 12]

    elif num_quant == 9:
        quant_idxs = [1, 1, 1,  2, 2, 2,  3, 3, 3,  4, 4, 4,   5, 5,  6, 6,  7, 7, 7,  8, 8,  9, 9]

    elif num_quant == 6:
        quant_idxs = [1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5,  6, 6, 6, 6]

    elif num_quant == 5:
        quant_idxs = [1, 1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, 5, 5]

    elif num_quant == 4:
        quant_idxs = [1, 1, 1, 1, 1,  2, 2, 2, 2, 2, 2,  3, 3, 3, 3, 3, 3, 3,   4, 4, 4, 4, 4]

    elif num_quant == 3:
        quant_idxs = [1, 1, 1, 1, 1, 1, 1,   2, 2, 2, 2, 2, 2, 2, 2,   3, 3, 3, 3, 3, 3, 3, 3]

    elif num_quant == 2:
        quant_idxs = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    elif num_quant == 1:
        quant_idxs = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # all quantizers are the same
    else:
        raise ValueError('unsupported number of quantizations')

    return quant_idxs


def getFixedVGGQuantIdxs(num_layers=16, num_quant=16):
    if num_quant == 16:
        quant_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] # all quantizers unique

    elif num_quant == 12:
        quant_idxs = [1, 1,  2, 2,  3, 3,  4, 4,  5, 6, 7, 8, 9, 10, 11, 12]

    elif num_quant == 9:
        quant_idxs = [1, 1,  2, 2,  3, 3,  4, 4,  5, 5,  6, 6,  7, 7,  8,  9]

    elif num_quant == 6:
        quant_idxs = [1, 1, 1,  2, 2, 2,  3, 3, 3,  4, 4, 4,  5, 5,  6, 6]

    elif num_quant == 3:
        quant_idxs = [1, 1, 1, 1, 1, 1,  2, 2, 2, 2, 2,  3, 3, 3, 3, 3]

    elif num_quant == 1:
        quant_idxs = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # all quantizers are the same
    else:
        raise ValueError('unsupported number of quantizations')

    assert(len(quant_idxs) == num_layers)

    return quant_idxs


def getEvenlySpacedQuantIdxs(num_layers, num_quant, pre_skip=0, post_skip=0):
    num_layers_use = num_layers - pre_skip - post_skip
    num_quant = min(num_quant, num_layers_use)

    eps = 1e-10
    layer_ids = np.linspace(0.5 + eps, num_quant+0.5-eps, num_layers_use)
    quant_idxs = np.round(layer_ids).astype(np.int32)
    assert quant_idxs.min() == 1
    assert quant_idxs.max() == num_quant

    quant_idxs_list = [None]*pre_skip + quant_idxs.tolist() + [None]*post_skip
    return quant_idxs_list


