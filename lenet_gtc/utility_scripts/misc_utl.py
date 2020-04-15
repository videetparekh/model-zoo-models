import os
import platform
import numpy as np
from colorama import Fore, Back
import glob
import shutil
from datetime import datetime
import time
from functools import wraps
import h5py
from collections import OrderedDict

def factorize_strs(str_list, delim='/', return_common_str_only=False):
    if len(str_list) == 0:
        return ''
    elif len(str_list) == 1:
        return str_list[0]

    str_list_splits = [ s.split(delim) for s in str_list  ]
    n_splits = [ len(lst) for lst in str_list_splits]
    min_n_splits = min(n_splits)

    common = ''
    split_idx_match = 0
    for split_idx in range(min_n_splits):

        strs_i = [s[split_idx] for s in str_list_splits]
        all_equal = all(s == strs_i[0] for s in strs_i)

        if all_equal:
            common += delim + strs_i[0]
            split_idx_match = split_idx
        else:
            break

    if len(common) > 0:
        common = common[1:]
    uncommon_strs = [delim.join(s_list[split_idx_match+1:]) for s_list in str_list_splits ]

    if return_common_str_only:
        final_str = common
    else:
        final_str = common + ' + ' + str(uncommon_strs)

    return final_str


def concat_detect_repeats(str_list_in):
    if not isinstance(str_list_in, list):
        raise ValueError('Expect list of strings as input')
    if len(str_list_in) == 0:
        return ''

    str_list = [str(x) for x in str_list_in]

    i = 1
    count = 1
    final_str = str_list[0]
    while i < len(str_list):
        if str_list[i] == str_list[i - 1]:
            count += 1
        else:
            final_str += (' x%d' % count if count > 1 else '') + '; ' + str_list[i]
            count = 1
        i += 1

    if count > 1:
        final_str += (' x%d' % count)

    return final_str


def cprint(str, color=Fore.RESET, backcolor=Back.RESET):
    print(color + backcolor + str + Fore.RESET + Back.RESET)


def onLaptop():
    return platform.node() == 'ziskinda-7730'


def models_root():
    models_dir = os.getenv('MODELS_DIR')

    if models_dir is None:
        if platform.node() == 'ziskinda-7730':
            models_dir = 'F:/models/'

        elif platform.node() in ['dare-dev', 'buckbeak'] or 'diva' in platform.node():
            models_dir = '/home/local/SRI/e28773/AVI/bitnet/models/'


    if models_dir is None and os.getenv('IN_DOCKER') is not None:
        models_dir = '/root/data/models/' # for docker image.

    if not bool(models_dir):
        raise ValueError('please define a MODELS_DIR environment variable')

    return models_dir

def data_root_deprecated():

    data_dir = os.getenv('DATA_DIR')

    if platform.node() == 'ziskinda-7730':
        data_dir = 'F:/'

    if data_dir is None and os.getenv('IN_DOCKER') is not None:
        data_dir = '/root/data/' # for docker image.

    if data_dir is None:
        raise ValueError('Please specify root data dir by '
                         'defining the environment variable "DATA_DIR"')

    return data_dir


def datasets_root():

    datasets_dir = '/home/gtc-tensorflow/Data/'#os.getenv('DATASETS_DIR')

    if datasets_dir is None:
        if platform.node() == 'ziskinda-7730':
            datasets_dir = 'F:/datasets/'
        elif platform.node() == 'dare-dev':
            datasets_dir = '/data/DataSets/'
        elif 'diva' in platform.node():
            datasets_dir = '/data/diva-2/avi/datasets/'
        elif platform.node() == 'hedwig':
            datasets_dir = '/data1/'


    if datasets_dir is None and os.getenv('IN_DOCKER') is not None:
        datasets_dir = '/root/data/datasets/' # for docker image.

    if datasets_dir is None or len(datasets_dir) == 0:
        raise ValueError('Please specify root datasets folder  by '
                         'defining the environment variable "DATASETS_DIR"')

    if not datasets_dir.endswith('/'):
        datasets_dir += '/'

    return datasets_dir

def fileOlderThan(filename, date_threshold):
    if not os.path.exists(filename):
        return False

    last_modification_time = os.path.getmtime(filename)
    if isinstance(date_threshold, tuple):
        time_threshold = datetime(*date_threshold).timestamp()
    elif isinstance(date_threshold, str) and os.path.exists(date_threshold):
        comparison_file = date_threshold
        time_threshold = os.path.getmtime(comparison_file)
    else:
        raise ValueError('Unsupported date format')

    return last_modification_time < time_threshold



def repo_root():
    if platform.node() == 'ziskinda-7730':
        return 'F:/SRI/bitnet/gtc-tensorflow/'
    elif os.getenv('IN_DOCKER'):
        return '/root/gtc-tensorflow/'
    elif platform.node() in ['dare-dev', 'hedwig', 'buckbeak'] or 'diva' in platform.node():
        return '/home/local/SRI/e28773/AVI/bitnet/gtc-tensorflow/'
    else:
        return '.'
        #raise ValueError('Unrecognized system')

def pretrained_weights_root():
    cache_dir = os.path.join(os.path.expanduser('~'), '.keras')
    weights_dir = os.path.join(cache_dir, 'models') + '/'

    return weights_dir

def trained_weights_root():
    return models_root()


def remove_None_values(d):
    return { k: v for k, v in d.items() if v is not None}


def get_small_num_str(x):

    if x == 0:
        return '0'

    is_pow_of_10 = np.round( np.log10(x) ) == np.log10(x)
    if is_pow_of_10:
        x_str = 'e%d' % np.log10(x)
    else:
        x_str = ('%.3g' % x).replace('e-0', 'e-')

    return x_str


#lambda_weights = namedtuple('lambda_weights', ['hp', 'lp', 'distillation_loss', 'bit_loss'])
#lambda_weights.__new__.__defaults__ = (1, None, None, None)


def abbrev_list(X, sep='_', maxRunBeforeAbbrev=2):
    '''
    concatenates non-sequential items into a list
        1,4,9 -->1_4_9

    abbreviates sequences (containing at least 3 elements):
        1,2,3,4,5         --> 1t5     -- like MATLAB's  1:5
        1,3,5,7,9         --> 1t2t9   -- like MATLAB's  1:2:9
        1,1.5,2,2.5,3     --> 1h3     -- "h" = special separator for steps of 0.5  (*h*alf)
        1,1.25,1.5,1.75,2 --> 1q2     -- "q" = special separator for steps of 0.25 (*q*uarter)
        0,5,10,15,20      --> 0f20    -- "f" = special separator for steps of 5    (*f*ive)
        0,10,20,30,40     --> 0d40    -- "d" = special separator for steps of 10   (*d*ecade)

    abbreviates repeated elements
        1,1,1,1  --> 1r4        -- like a special separator for steps of 0

    combines mixes of sequences:
        1,2,3,4, 8,8.5,9, 15,15,15, 20  --> 1t4_8h9_15r3_20

    '''

    if maxRunBeforeAbbrev < 0:
        maxRunBeforeAbbrev = 1e10


    abbrevSepValues = {1:'t', 0.5:'h', 0.25:'q', 5:'f', 10:'d'}

    useHforHalfValues = True

    final_str = ''


    if isinstance(X, (list, tuple)):
        if len(X) == 0:
            return ''

        L = len(X)
        # maxN = math.min(maxN or #X, #X)

        X_str = [str(x) for x in X]
        curIdx = 0
        final_str = X_str[0]
        while curIdx < L-1:  # maxN
            runLength = 0
            initDiff = X[curIdx+1] - X[curIdx]
            curDiff = initDiff
            while (curIdx+runLength < L-1) and (curDiff == initDiff):
                runLength += 1
                if curIdx + runLength < L-1:
                    curDiff = X[curIdx+runLength+1] - X[curIdx+runLength]


            # print('run = ', runLength)
            if runLength >= maxRunBeforeAbbrev:
                # print('a');
                # print( 't' .. X[curIdx+runLength] )
                if initDiff == 0:
                    final_str += 'r' + str(runLength+1)

                else:
                    abbrevSep = None
                    for diffVal,diffSymbol in abbrevSepValues.items():
                        if initDiff == diffVal:
                            abbrevSep = diffSymbol

                    if abbrevSep is None:
                        # print(initDiff)
                        abbrevSep = 't%st' % str(initDiff)

                    final_str += abbrevSep + X_str[curIdx+runLength]


                curIdx = curIdx + runLength+1
            else:
                # print('b');
                # print( table.concat(X, sep, curIdx, curIdx+runLength) )
                if (runLength > 0):
                    final_str += sep + sep.join(X_str[slice(curIdx + 1, curIdx + runLength+1)])

                curIdx = curIdx + runLength+1

            if curIdx < L:
                final_str += sep + X_str[curIdx]



    elif isinstance(X, (int, float)):
        final_str = str(X)
    else:
        raise ValueError('Unhandled case type(X) = %s ' % type(X))

    final_str = final_str.replace('-', 'n')

    if useHforHalfValues:
        final_str = final_str.replace('.5', 'H')

    return final_str



def clear_out_empty_models_folders(model_subfolder, test_mode=True):

    if isinstance(model_subfolder, list):
        for sub_i, model_subfolder_i in enumerate(model_subfolder):
            print(' *** Model sub folder %d/%d *** ' % (sub_i+1, len(model_subfolder)))
            clear_out_empty_models_folders(model_subfolder_i, test_mode=test_mode)
        return

    assert isinstance(model_subfolder, str), "Expected string"
    print(' ** Clearing out model subfolder %s ** ' % model_subfolder)

    path = models_root() + model_subfolder + '/'
    if not os.path.exists(path):
        print(' *** No path %s exists ***' % path)
        return
    all_files = [os.listdir(path)]
    subdirs = [s for s in os.listdir(path) if os.path.isdir(path + s)]
    print('Found %d sub-folders (out of %d total items) in %s' % (len(subdirs), len(all_files), path))

    if onLaptop():
        max_filename_length = 260
    else:
        max_filename_length = 1000

    for j, subdir in enumerate(subdirs):
        subdir_full = path + subdir
        training_folders = glob.glob(subdir_full + '/training*') + glob.glob(subdir_full + '/20*')
        training_folders = [s for s in training_folders if os.path.isdir(s)]

        cprint('\nDir %d / %d : %s (%d training folders)' % (j+1, len(subdirs), subdir, len(training_folders)), color=Fore.CYAN )


        for k,training_folder in enumerate(training_folders):
            #print('training_folder', training_folder)
            csv_files = glob.glob(training_folder + '/*.csv')
            tot_csv_size_kb = 0
            for ci, csv_file in enumerate(csv_files):
                if len(csv_file) >= max_filename_length:
                    cprint('\n File name is too long (%d chars) : %s \n skipping this folder.' % (len(csv_file), csv_file), color=Fore.MAGENTA)
                    continue

                fsize = os.path.getsize(csv_file)
                tot_csv_size_kb += fsize / 1024

            txt_files = glob.glob(training_folder + '/*.txt')
            tot_text_size_kb = 0
            for ci, txt_file in enumerate(txt_files):
                fsize = os.path.getsize(txt_file)
                tot_text_size_kb += fsize / 1024

            have_quant_log_csv_file = any(['quant.csv' in s for s in csv_files])
            min_csv_kb = 0
            if have_quant_log_csv_file:
                min_csv_kb = 3
            to_delete = tot_csv_size_kb <= min_csv_kb and tot_text_size_kb <= 1


            color = Fore.RED if  to_delete else Fore.GREEN
            cprint('    Training folder %d/%d. txt_size=%.1f kb csv_size = %.1f kb. [have-quant=%s] %s ' % (
                    k+1, len(training_folders), tot_text_size_kb, tot_csv_size_kb, have_quant_log_csv_file, 'DELETE' if to_delete else ' (KEEP)'), color=color)

            if not test_mode and to_delete:
                cprint('    DELETING %s' % training_folder, color=Fore.RED)
                shutil.rmtree(training_folder)

            pass

        training_folders = glob.glob(subdir_full + '/training*') + glob.glob(subdir_full + '/20*')
        training_folders = [s for s in training_folders if os.path.isdir(s)]

        all_stuff = glob.glob(subdir_full + '/*')

        if len(training_folders) == 0:
            if len(all_stuff) == 0:
                cprint('    Dir %d/%d (%s) is empty. deleting...' % (j+1, len(subdirs), subdir), color=Fore.RED)
                if not test_mode:
                    cprint('    DELETING DIRECTORY %s' % subdir, color=Fore.RED)
                    os.rmdir(subdir_full)
            else:
                cprint('    Dir %d/%d (%s) is not empty... CANNOT delete...' % (j + 1, len(subdirs), subdir), color=Fore.MAGENTA)




def retry(ExceptionToCheck, tries=4, delay=3, backoff=2, logger=None):
    """Retry calling the decorated function using an exponential backoff.

    http://www.saltycrane.com/blog/2009/11/trying-out-retry-decorator-python/
    original from: http://wiki.python.org/moin/PythonDecoratorLibrary#Retry

    :param ExceptionToCheck: the exception to check. may be a tuple of
        exceptions to check
    :type ExceptionToCheck: Exception or tuple
    :param tries: number of times to try (not retry) before giving up
    :type tries: int
    :param delay: initial delay between retries in seconds
    :type delay: int
    :param backoff: backoff multiplier e.g. value of 2 will double the delay
        each retry
    :type backoff: int
    :param logger: logger to use. If None, print
    :type logger: logging.Logger instance
    """
    def deco_retry(f):

        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except ExceptionToCheck as e:
                    msg = "%s, Retrying in %d seconds..." % (str(e), mdelay)
                    if logger:
                        logger.warning(msg)
                    else:
                        print(msg)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)

        return f_retry  # true decorator

    return deco_retry


def h5py_to_dict(g):
    d = OrderedDict()
    for k, v in h5py_walk(g):
        d[k] = v
    return d


def h5py_walk(g, prefix=''):
    for key in g.keys():
        item = g[key]
        path = '{}/{}'.format(prefix, key)
        if isinstance(item, h5py.Dataset):  # test for dataset
            if not item.shape:
                # scalar
                item_val = item[()]
            else:
                item_val = item[:]
            yield (path, item_val)
        elif isinstance(item, h5py.Group):  # test for group (go down)
            yield from h5py_walk(item, path)


if __name__ == "__main__":
    test_abbrev_list = False
    if test_abbrev_list:
        print(abbrev_list([1, 4, 9]))
        print(abbrev_list([1, 2, 3, 4, 5]))
        print(abbrev_list([1, 3, 5, 7, 9]))

        print(abbrev_list([1, 1.5, 2, 2.5, 3]))
        print(abbrev_list([1, 1.25, 1.5, 1.75, 2]))
        print(abbrev_list([0, 5, 10, 15, 20]))

        print(abbrev_list([1, 1, 1, 1]))
        print(abbrev_list([1, 2, 3, 4, 8, 8.5, 9, 15, 15, 15, 20]))

    test_factorize_strs = False
    if test_factorize_strs:
        str_list = [
        'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_4_Conv2d_1_1x1_64/BatchNorm/gamma:0:[64]',
        'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_4_Conv2d_1_1x1_64/BatchNorm/beta:0:[64]',
        'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_4_Conv2d_1_1x1_64/BatchNorm/moving_mean:0:[64];',
        'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_4_Conv2d_1_1x1_64/BatchNorm/moving_variance:0:[64]'
        ]

        s_factorized = factorize_strs(str_list)
        print(s_factorized)

        str_list2 = ['(None, None, 21)', '(None, None, 21)', '(None, None, 21)',   '(None, None, 4)', '(None, None, 4)']
        s_concat = concat_detect_repeats(str_list2)
        print(s_concat)

        print('Done!')

    test_toList = False
    if test_toList:

        '''
        concatenates non-sequential items into a list
            1,4,9 -->1_4_9
    
        abbreviates sequences (containing at least 3 elements):
            1,2,3,4,5         --> 1t5     -- like MATLAB's  1:5
            1,3,5,7,9         --> 1t2t9   -- like MATLAB's  1:2:9
            1,1.5,2,2.5,3     --> 1h3     -- "h" = special separator for steps of 0.5  (*h*alf)
            1,1.25,1.5,1.75,2 --> 1q2     -- "q" = special separator for steps of 0.25 (*q*uarter)
            0,5,10,15,20      --> 0f20    -- "f" = special separator for steps of 5    (*f*ive)
            0,10,20,30,40     --> 0d40    -- "d" = special separator for steps of 10   (*d*ecade)
    
        abbreviates repeated elements
            1,1,1,1  --> 1r4        -- like a special separator for steps of 0
    
        combines mixes of sequences:
            1,2,3,4, 8,8.5,9, 15,15,15, 20  --> 1t4_8h9_15r3_20
    
        '''


    networks_clear = ['mobilenet_v2-keras', 'resnet18_ic']
    if onLaptop():
        machine_sub_folders = ['', '_dare-dev', '_buckbeak', '_diva']
        all_networks_clear =[ [m + '/' + net for m in machine_sub_folders] for net in  networks_clear ]
        networks_clear = sum(all_networks_clear, [])


    clear_out_folders = False
    if clear_out_folders:
        clear_out_empty_models_folders(networks_clear, test_mode=True)
        #clear_out_empty_models_folders('resnet18_ic', test_mode=False)



    print('Done!!')
    a = 1

