import keras
import os
import time
import numpy as np

import keras.backend as K
import warnings
from collections import OrderedDict
from colorama import Fore

from utility_scripts.weights_readers import ModelWeightsReader
from utility_scripts import misc_utl as utl


class DetailedCSVLogger(keras.callbacks.CSVLogger):
    # Same as a regular CSV Logger, with some modifications:
    # - Logs the time of each epoch, as well as how many batches & samples it included
    # - outputs keys in specified order
    # - also logs the learning rate.

    # Inherits: def __init__(self, filename, separator=',', append=False):
    def __init__(self, filename, separator=',', append=False,
                 keys_order=None, add_lambdas=False, save_copy_to_txt=True):
        self.keys_order = keys_order
        self.keys_final = None
        self.batch_count = 0
        self.samples_count = 0
        self.add_lambdas = add_lambdas
        self.save_copy_to_txt = save_copy_to_txt
        self.val_line_fmt = None

        if save_copy_to_txt:
            self.txt_filename = filename.replace('.csv', '_csv.txt')
            if append and os.path.exists(self.txt_filename):
                open_mode = 'a'
            else:
                open_mode = 'w'

            with open(self.txt_filename, open_mode) as fid:
                pass


        super(DetailedCSVLogger, self).__init__(filename, separator, append)

    # Inherits: def on_train_begin(self, logs=None):
    def __str__(self):
        return 'DetailedCSVLogger (%s)' % os.path.basename(self.filename)


    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_begin_time = time.time()
        self.epoch_batch_count = 0
        self.epoch_samples_count = 0

    def on_batch_end(self, batch, logs=None):
        size = logs['size']
        self.epoch_batch_count += 1
        self.epoch_samples_count += size

    def get_lambdas(self, keys_only=False):
        lambda_weights = self.model.gtcModel.lambda_weights
        abbrev = lambda_weights.get_abbrev()
        all_keys = []
        all_lambdas = OrderedDict()
        for k in lambda_weights.get_keys():
            k_abbrev = 'lam_' + abbrev[k]
            all_keys.append(k_abbrev)
            if not keys_only:
                val = float(K.get_value(getattr(lambda_weights, k)))
                all_lambdas[k_abbrev] = val

        if keys_only:
            return all_keys
        else:
            return all_lambdas

    def on_epoch_end(self, epoch, logs=None):
        epoch_duration = time.time() - self.epoch_begin_time

        if self.keys_final is None:
            # First time only: determine key order.
            log_keys = list(logs.keys())
            ordered_keys = self.keys_order if self.keys_order is not None else log_keys

            keys_reordered = [k for k in ordered_keys if k in log_keys and not k.startswith('val_')]

            # Add remaining keys not specified in keys_order
            remaining_keys = sorted([k for k in log_keys if k not in keys_reordered and not k.startswith('val_')])
            keys_reordered.extend(remaining_keys)

            # Add validation keys in same order as original
            # Include exception for mAP which is calculated in a separate callback (and no val_ copy is made)
            val_keys_ordered = ['val_' + k for k in keys_reordered  if ('val_' + k in log_keys and 'mAP' not in k)]
            keys_reordered.extend(val_keys_ordered)
            assert len(keys_reordered) == len(log_keys), "missing or duplicate keys"
            assert sorted(keys_reordered) == sorted(log_keys), "keys mismatch"

            keys_reordered.insert(0, 'batches')
            keys_reordered.insert(1, 'samples')
            keys_reordered.insert(2, 'epoch_t')

            # Learning rate parameters/lambas (at the end)
            keys_reordered.append('learning_rate')
            if self.add_lambdas:
                lambda_keys = self.get_lambdas(keys_only=True)
                keys_reordered.extend(lambda_keys)
            self.keys_final = keys_reordered
            print('keys_reordered', keys_reordered)

        logs_final = logs.copy() #OrderedDict([(k, (logs[k] if k in logs else 0.0)) for k in self.keys_final])
        logs_final['batches'] = self.epoch_batch_count
        logs_final['samples'] = self.epoch_samples_count
        logs_final['epoch_t'] = epoch_duration
        logs_final['learning_rate'] = float(K.get_value(self.model.optimizer.lr))
        if self.add_lambdas:
            lambda_values = self.get_lambdas()
            logs_final.update(lambda_values)

        # Call the 'on_batch_end' of the parent class, which logs to the final
        self.keys = self.keys_final
        logs_final_ordered = OrderedDict([(k,logs_final[k]) for k in self.keys_final ])

        super(DetailedCSVLogger, self).on_epoch_end(epoch, logs_final_ordered)

        if self.save_copy_to_txt:
            self.write_logs_to_txtfile(epoch, logs_final_ordered)


    def write_logs_to_txtfile(self, epoch, logs):
        logs = logs.copy()
        log_keys = list(logs.keys())

        logs['epoch'] = epoch
        log_keys.insert(0, 'epoch')

        if epoch == 0 or self.val_line_fmt is None:

            key_name_lengths = [max(len(s), 10) for s in log_keys]
            key_name_digits = [0, 0] + [4]*(len(log_keys)-2)  # integers for epochs, #samples
            val_line_fmt = ' '.join([ ('%%%d.%df' % (l,p) ) for l,p in zip(key_name_lengths, key_name_digits)  ]) + '\n'
            self.val_line_fmt = val_line_fmt

            if epoch == 0:
                heading_line_fmt = ' '.join(['%' + str(n) + 's' for n in key_name_lengths]) + '\n'
                heading_str = heading_line_fmt % tuple(log_keys)
                with open(self.txt_filename, 'a+') as fid:
                    fid.write(heading_str)


        vals = tuple(logs[k] for k in log_keys)
        vals_str = self.val_line_fmt % vals

        with open(self.txt_filename, 'a+') as fid:
            fid.write(vals_str)

    # Inherits: def on_train_end(self, logs=None):



class BatchCSVLogger(keras.callbacks.CSVLogger):

    """Callback that streams BATCH results to a csv file.

    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.

    # Example

    ```python
    csv_logger = CSVLogger('training.log')
    model.fit(X_train, Y_train, callbacks=[csv_logger])
    ```

    # Arguments
        filename: filename of the csv file, e.g. 'run/log.csv'.
        separator: string used to separate elements in the csv file.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    """

    def __init__(self, filename, separator=',', append=False,
                 keys_order=None, batch_log_freq=1):
        self.keys_order = keys_order
        self.keys_final = None
        self.batch_log_freq = batch_log_freq
        self.batch_count = 0
        self.samples_count = 0
        self.logs_cache = {}

        #self.display = display

        super(BatchCSVLogger, self).__init__(filename, separator, append)

    def __str__(self):
        return 'BatchCSVLogger (Log to %s every %d batches)' % (
            os.path.basename(self.filename), self.batch_log_freq)

    def on_epoch_begin(self, epoch, logs=None):
        # self.epoch_begin_time = time.time()
        self.epoch_id = epoch
        self.epoch_samples_count = 0
        self.logging_sample_count = 0
        self.mismatch_keys_warned = {}  # reset dict of logs .


    def on_batch_begin(self, batch, logs=None):
        self.batch_begin_time = time.time()

    def on_batch_end(self, batch, logs=None):
        logs = logs.copy()
        batch_size = logs.pop('size', 0)
        self.batch_count += 1

        self.samples_count += batch_size  # TOTAL number of samples seen
        self.epoch_samples_count += batch_size # Total number seen this epoch
        self.logging_sample_count += batch_size # Total in this logging period

        batch_duration = time.time() - self.batch_begin_time

        # epoch, batch, samples,  batch_t

        batch_tmp = logs.pop('batch')


        if self.keys_final is None:
            # First time only: determine key order.
            log_keys = list(logs.keys())
            ordered_keys = self.keys_order if self.keys_order is not None else log_keys

            keys_reordered = [k for k in ordered_keys if k in log_keys]
            remaining_keys = sorted([k for k in log_keys if k not in keys_reordered])
            keys_reordered.extend(remaining_keys)

            # logs_curvalue is the value for the current batch (calculated)
            # logs_cumsum accumulates over 'self.batch_log_freq' batches
            # logs_epoch_cumsum accumulates over the entire epoch (from keras)
            self.logs_curvalue = {k: 0 for k in keys_reordered}
            self.logs_cumsum = {k: 0 for k in keys_reordered}
            self.logs_epoch_cumsum = {k: 0 for k in keys_reordered}
            self.logs_cumaverage = {k: 0 for k in keys_reordered}

            keys_final = keys_reordered.copy()
            keys_final.insert(0, 'batch')
            keys_final.insert(1, 'samples')
            keys_final.insert(2, 'batch_t')
            keys_final.append('learning_rate')
            self.keys_final = keys_final
            self.keys_reordered = keys_reordered



        # Keep running total of all logs
        for k in self.keys_reordered:
            if k in logs:
                # Keras supplies the running mean. We want the instantaneous
                # values (curvalue), so we have to "undo" the running mean operation
                # by calculating the change since the previous running mean.

                # We use the self.epoch_samples_count and self.logs_epoch_cumsum
                # to undo the running average and get the instantaneous values
                # We then calculate our own running mean over the shorter
                # timescale of  self.batch_log_freq  batches.
                self.logs_curvalue[k] = np.float64(logs[k]) * (self.epoch_samples_count) - self.logs_epoch_cumsum[k]
                self.logs_epoch_cumsum[k] += self.logs_curvalue[k]
                if not np.isnan(self.logs_epoch_cumsum[k]) and np.abs(self.logs_epoch_cumsum[k]) < 1e6:
                    calc_running_mean = self.logs_epoch_cumsum[k]/self.epoch_samples_count
                    if not np.isclose( calc_running_mean, logs[k]):
                        if k not in self.mismatch_keys_warned:
                            self.mismatch_keys_warned[k] = True
                            warnings.warn('Calculated running mean (%g) and actual running mean (%g) are different' % (
                                calc_running_mean, logs[k]), UserWarning)  # try to reproduce running mean

                self.logs_cumsum[k] += self.logs_curvalue[k]
                # For debugging, keep track of cumulative average over shorter timescale
                self.logs_cumaverage[k] = self.logs_cumsum[k] / self.logging_sample_count

                #self.logs_cache[k] += logs[k] * batch_size

        if self.batch_count % self.batch_log_freq == 0:

            logs_final = {k: v/self.logging_sample_count for k,v in self.logs_cumsum.items() }

            # reset the running average: set counter to zero, and sums to 0
            self.logging_sample_count = 0
            for k in self.logs_cumsum.keys():
                self.logs_cumsum[k] = 0

            if False:
                metrics_log = ''
                for k in self.keys_reordered:
                    val = logs_final[k]
                    if abs(val) > 1e-3:
                        metrics_log += ' - %s: %.4f' % (k, val)
                    else:
                        metrics_log += ' - %s: %.4e' % (k, val)
                print('step: {}/{} ... {}'.format(self.batch_count,
                                                  self.params['steps'],
                                                  metrics_log))

            #logs_final = OrderedDict([(k, (logs[k] if k in logs else 0.0)) for k in self.keys_final])
            logs_final['batch'] = self.batch_count
            logs_final['samples'] = self.samples_count
            logs_final['batch_t'] = batch_duration
            logs_final['learning_rate'] = float(K.get_value(self.model.optimizer.lr))

            self.keys = self.keys_final
            logs_final_ordered = OrderedDict([(k, logs_final[k]) for k in self.keys_final])
            super(BatchCSVLogger, self).on_epoch_end(self.epoch_id, logs_final_ordered)


    def on_epoch_end(self, epoch, logs=None):
       self.logs_curvalue = {k: 0 for k in self.keys_reordered}
       self.logs_cumsum = {k: 0 for k in self.keys_reordered}
       self.logs_epoch_cumsum = {k: 0 for k in self.keys_reordered}



class QuantizationHistoryLogger(keras.callbacks.CSVLogger):


    def __init__(self, filename, separator=',', append=False,
                 log_quant_params=True, log_bits_per_layer=True,
                 log_loss_lambdas=True,
                 quantized_model=None):
        # For multi-gpu training, pass in the original template (single-gpu)
        # model in as the 'quantized model'.

        self.log_quant_params = log_quant_params
        self.log_bits_per_layer = log_bits_per_layer
        self.log_loss_lambdas = log_loss_lambdas
        self.quantized_model = quantized_model
        self.model_use = None
        super(QuantizationHistoryLogger, self).__init__(filename, separator, append)


    def __str__(self):
        return 'QuantizationHistoryLogger (%s)' % (
            os.path.basename(self.filename) )

    def on_train_begin(self, logs=None):
        # Call the parent class 'on_train_begin', which opens the log file.
        super(QuantizationHistoryLogger, self).on_train_begin(None)

        if self.quantized_model is None:
            self.model_use = self.model
        else:
            self.model_use = self.quantized_model

        self.model.stop_training = False  # on epoch end expects this variable to exist.

        # At beginning of training, log the current state of all the
        # quantization parameters, under epoch # "-1"
        logs = self.get_quant_variables()

        self.keys = list(logs.keys())
        self.on_epoch_end(-1, logs=logs)

    def on_epoch_end(self, epoch, logs=None):

        logs = self.get_quant_variables()
        super(QuantizationHistoryLogger, self).on_epoch_end(epoch, logs)


    def get_quant_variables(self):

        logs = OrderedDict()
        # Log the initial states of the quantization variables under 'epoch 0'
        if self.log_quant_params:
            all_quant_vars = self.model_use.gtcModel.get_quantization_variables(evaluate=True, abbrev=True)
            logs.update( all_quant_vars )

            # add summary statistics
            quant_var_keys = all_quant_vars.keys()
            # average slope:
            slope_vals = [ s for k,s in all_quant_vars.items() if '_slp' in k]
            if bool(slope_vals):
                logs['mean_slope'] = np.mean(slope_vals)

            intercept_vals = [ i for k,i in all_quant_vars.items() if '_int' in k]
            if bool(intercept_vals):
                logs['mean_intercept'] = np.mean(intercept_vals)

            linear_slope_vals = [ i for k,i in all_quant_vars.items() if '_Lslp' in k]
            if bool(linear_slope_vals):
                logs['mean_lin_slope'] = np.mean(linear_slope_vals)


        if self.log_bits_per_layer:
            all_bits_per_layer = self.model_use.gtcModel.get_bits_per_layer(evaluate=True, abbrev=True)
            logs.update ( all_bits_per_layer )

            all_bits_per_layer_vals = list(all_bits_per_layer.values())
            mean_bits_per_layer = np.mean(all_bits_per_layer_vals)
            logs['mean_bits_per_layer'] = mean_bits_per_layer

        if self.log_loss_lambdas:
            lambda_weights = self.model_use.gtcModel.lambda_weights
            abbrev = lambda_weights.get_abbrev()
            for k in lambda_weights.get_keys():
                val = float(K.get_value( getattr(lambda_weights, k )))
                k_abbrev = abbrev[k]
                logs['lam_' + k_abbrev] = val

            # add summary statistics

        # log all the number of bits for each layer:

        return logs








class AdjustLambdaCallback(keras.callbacks.Callback):
    """ Adjust one of the lambdas when a certain condition applies.
    Currently supported:
        (1) increase distillation loss lambda if hp/lp accuracies (or losses)
            are not converging to each other fast enough
        (2) increase bit loss lambda if bits_per_layer is not converging to a
            target value fast enough.

    Adapted from the keras callback ReduceLROnPlateau

    # Example

    ```python
    adjust_distillation_cb = AdjustLambdaCallback(
        monitor='lp_acc', target='hp_acc', min_delta=2.0
        lambda_name='distillation_loss', factor=2.0, max_lambda=0.1,
        patience=3, cooldown=3)

    adjust_bitloss_cb = AdjustLambdaCallback(
        monitor='bits_per_layer', target=4.0, min_delta=0.1,
        lambda_name='bit_loss', factor=1.4,  max_lambda=0.1,
        patience=3, cooldown=3)
    model.fit(X_train, Y_train, callbacks=[adjust_distillation_cb, adjust_bitloss_cb])

    ```

    # Arguments
        monitor: quantity to be monitored.
        target: quantity to compare with 'monitor'
        lambda_name: the name of the lambda loss (eg. 'distillation_loss')
        factor: factor by which the lambda will
            be increased. new_lambda = lambda * factor
        patience: number of epochs with no improvement
            after which lambda will be increased.
        verbose: int. 0: quiet, 1: update messages.
        min_delta: threshold for measuring improvements in the monitor
            value, to only focus on significant changes.
        cooldown: number of epochs to wait before resuming
            normal operation after lambda has been increased.
        max_lambda: upper bound on the lambda.
    """

    def __init__(self, monitor='lp_acc', target='hp_acc', min_delta=2.0,
                 lambda_name='distillation_loss', factor = 2.0, max_lambda=1.0,
                 prefer_validation_data=True,
                 patience=3, cooldown=3, verbose=0, color=None):
        super(AdjustLambdaCallback, self).__init__()


        self.monitor = monitor
        self.target = target
        self.min_delta = min_delta
        self.prefer_validation_data = prefer_validation_data
        self.lambda_name = lambda_name
        if factor <= 1.0:
            raise ValueError('Factor must be > 1.0.')
        self.factor = factor
        self.max_lambda = max_lambda

        self.patience = patience
        self.cooldown = cooldown
        self.verbose = verbose

        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0

        self.mode = 'auto'
        self.monitor_op = None
        self._reset()
        if color is None:
            if 'distillation' in lambda_name:
                color = Fore.MAGENTA
            elif 'bit' in lambda_name:
                color = Fore.GREEN
            else:
                color = Fore.CYAN

        self.color = color
        self.print = lambda s : utl.cprint(s, color=self.color)

    def __str__(self):
        return 'AdjustLambdaCallback (%s->%s. minDelta=%g. Adjust %s, ' \
               'factor=%g. patience=%d, cooldown=%d)' % (
            self.monitor, str(self.target), self.min_delta,
            self.lambda_name, self.factor,
            self.patience, self.cooldown)


    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        # Monitor_op: return True if things improving ok,
        #             return False if we need to take action and adjust lambda
        if (self.mode == 'min' or
           (self.mode == 'auto' and 'acc' not in self.monitor)):

            # For a loss metric, trying to minimize the 'monitor' value, down to 'target',
            # (e.g  bits_per_layer, down to target_bits)
            # so presumably monitor > target, and thus monitor-target > 0:
            #           thus  current = monitor-target > 0. (positive values, hopefully decreasing)
            #  OK (=return True) if  (current < best_prev - delta)          [improving_op]
            #                 OR if   monitor < target + delta              [reached_target_op]
            self.improving_op = lambda a, b: np.less(a, b - self.min_delta)
            self.reached_target_op = lambda a, b: np.less(a, b + self.min_delta)
            #self.monitor_op = lambda val, target:  val - target < self.min_delta
            self.best = np.Inf
        else:
            # For accuracy metric, presumably want to maximize value
            # (e.g  increasing lp_acc to match target hp_acc
            # So presumably  monitor < target. (e.g. lp_acc < hp_acc)
            #           thus current = monitor-target < 0.
            #           (negative values, hopefully increasing, approaching 0)
            # So OK(=return True)  if ( current > prev_best + delta)          [improving_op]
            #                   OR if   monitor > target - delta              [reached_target_op]
            self.improving_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.reached_target_op = lambda a, b: np.greater(a, b - self.min_delta)
            #self.monitor_op = lambda val, target: target - val  < self.min_delta
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0


    def on_train_begin(self, logs=None):
        self._reset()
        assert hasattr(self.model.gtcModel, 'lambda_weights'), \
            "GTCModel does not have any lambda weights to adjust"


    def get_value(self, param, logs):
        if isinstance(param, (float, int, np.number)):
            return float(param)

        elif isinstance(param, str):
            if param in ['bits', 'bits_per_layer']:
                param = 'bits_per_layer_loss'

            #val_param = 'val_' + param
            #if self.prefer_validation_data and val_param in logs:
            #    param = val_param

            if param in logs:
                return logs[param]

        else:
            return None


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

        monitor_val = self.get_value(self.monitor, logs)
        target_val = self.get_value(self.target, logs)
        current = monitor_val - target_val

        if monitor_val is None or target_val is None:
            warnings.warn(
                'AdjustLambdaCallback conditioned on metrics'
                ' `%s` and `%s`, which are not both available.'
                '. Available metrics are: %s' %
                (str(self.monitor), str(self.target), ','.join(list(logs.keys()))), RuntimeWarning
            )

        else:
            is_improving = self.improving_op(current, self.best)
            reached_target = self.reached_target_op(monitor_val, target_val)

            message_prefix = '    >> Epoch %05d: AdjustLambdaCallback(%s):' % (epoch + 1, self.lambda_name)
            target_str = str(self.target)
            if str(target_val) != target_str:
                target_str += ' (%s)' % str(target_val)

            if self.in_cooldown():
                if self.verbose:
                    self.print(message_prefix + '<In cooldown: remaining epochs = %d>' % self.cooldown_counter)
                self.cooldown_counter -= 1
                self.wait = 0

            if is_improving or reached_target:
                if self.verbose > 0:
                    if reached_target:
                        self.print(message_prefix + " OK: %s [%g] is within delta=%.g of target %s" % (self.monitor, monitor_val, self.min_delta, target_str))
                    elif is_improving:
                        self.print(message_prefix + " OK: %s [%g] is improving. Prev diff: %g. Current diff: %s" % (self.monitor, monitor_val, self.best, current))
                self.best = current
                self.wait = 0


            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    lambda_variable= getattr(self.model.gtcModel.lambda_weights, self.lambda_name)
                    old_lambda = float(K.get_value(lambda_variable))
                    #old_lambda = float(K.get_value(self.model.lambdas.distillation_loss))
                    if old_lambda == 0:
                        old_lambda = 1e-6
                        warnings.warn('Lambda started out at 0.0. Setting to a small value (%g) so annealing can begin.' % old_lambda, UserWarning)


                    if old_lambda < self.max_lambda:
                        new_lambda = old_lambda * self.factor
                        new_lambda = min(new_lambda, self.max_lambda)
                        K.set_value(lambda_variable, new_lambda)
                        if self.verbose > 0:

                            self.print(message_prefix +
                                  'Current diff between %s (%g) and target of %s is %g. Best diff over last %d epochs is %.1f '
                                  '(Thus, no improvement by at least delta=%.1f). Increasing %s lambda from %g to %g. (multiplied by factor of %g)' % (
                                      self.monitor, monitor_val, target_str, current,
                                      self.patience, self.best,  self.min_delta,
                                      self.lambda_name, old_lambda, new_lambda, self.factor))

                        self.cooldown_counter = self.cooldown
                        self.wait = 0
                else:
                    if self.verbose > 0:
                        self.print(message_prefix + " Waiting for %s to improve [currently at %g]  [%d/%d]" % (
                            self.monitor, monitor_val, self.wait, self.patience))

    def in_cooldown(self):
        return self.cooldown_counter > 0






tbl_metrics_abbrev = {
    'epoch_time': 'epoch_t',
    '_time_av': '',
    'corrcoeff': 'cc',
    'mean_absolute_error': 'mae',
    'mean_squared_error': 'mse',
    'ssimcoeff': 'ssim',
    'pose_deg_err': 'pose_err',
    'val_': 'VAL_',
}

def metric_abbrev(metric_name):
    for k in tbl_metrics_abbrev.keys():
        if k in metric_name:
            metric_name = metric_name.replace(k,tbl_metrics_abbrev[k])
    return metric_name



class TrainingHistoryLogger(keras.callbacks.Callback):

    def __init__(self, log_file, append=True):
        self.log_file = log_file
        if append and os.path.exists(log_file):
            open_mode = 'a'
        else:
            open_mode = 'w'

        with open(log_file, open_mode) as fid:
            pass
        self.written_headers = False
        super(TrainingHistoryLogger, self).__init__()

    def __str__(self):
        return 'TrainingHistoryLogger (%s)' % os.path.basename(self.log_file)

    def on_train_begin(self, logs=None):
        self.losses = []
        self.val_losses = []
        self.epoch_durations = []

    def on_train_end(self, logs=None):
        return

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_begin_time = time.time()
        return

    def on_epoch_end(self, epoch, logs=None):
        logs = logs.copy()
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        epoch_duration = time.time() - self.epoch_begin_time

        # Handle cases where validation_freq > 1, and don't get validation data
        # on the first epoch. to keep the data file columns uniform,
        # we add some made-up zero validation data to fill the empty columns
        # until the actual validation data arrives.

        log_keys = sorted(list(logs.keys()))
        have_validation_keys = all(['val_' in k or 'val_' + k in log_keys for k in log_keys])
        if not have_validation_keys:
            new_log_keys = []
            for k in log_keys:
                new_key = 'val_' + k
                logs[new_key] = 0.0
                new_log_keys.append(new_key)
            log_keys.extend(new_log_keys)

        # Add epoch id & epoch duration
        log_keys.insert(0, 'epoch')
        log_keys.insert(1, 'epoch_time')
        logs['epoch'] = epoch + 1
        logs['epoch_time'] = epoch_duration

        log_keys_abbrev = [metric_abbrev(k) for k in log_keys]
        vals = tuple([logs[fld] for fld in log_keys])
        cur_n_values = len(vals)

        if not self.written_headers or cur_n_values != self.n_values:
            key_name_lengths = [max(len(s), 10) for s in log_keys_abbrev]
            key_name_digits = [0] + [4]*(len(log_keys)-1)
            val_line_fmt = ' '.join([ ('%%%d.%df' % (l,p) ) for l,p in zip(key_name_lengths, key_name_digits)  ]) + '\n'
            heading_line_fmt = ' '.join(['%' + str(n) + 's' for n in key_name_lengths]) + '\n'

            heading_str = heading_line_fmt % tuple(log_keys_abbrev)
            with open(self.log_file, 'a+') as fid:
                fid.write(heading_str)

            self.n_values = cur_n_values


            self.val_line_fmt = val_line_fmt
            self.written_headers = True

        vals_str = self.val_line_fmt % vals

        with open(self.log_file, 'a+') as fid:
            fid.write(vals_str)

        self.losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.epoch_durations.append(epoch_duration)
        #y_pred = self.model.predict(self.validation_data[0])
        return

    def on_batch_begin(self, batch, logs=None):
        return

    def on_batch_end(self, batch, logs=None):
        return



class ResetWeightsCallback(keras.callbacks.Callback):

    def __init__(self, weights_filename, verbose=True):
        self.weights_filename = weights_filename
        self.verbose = verbose
        self.reader = None
        self.use_reader = weights_filename.lower() == 'reader'
        super(ResetWeightsCallback, self).__init__()

    def __str__(self):
        return 'ResetWeightsCallback (%s)' % (os.path.basename(self.weights_filename))

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            if self.verbose:
                print('  ResetWeightsCallback: epoch %03d. Saving weights to %s' % (epoch, self.weights_filename))

            if self.use_reader:
                self.reader = ModelWeightsReader(self.model)
            else:
                self.model.save_weights(self.weights_filename)

            if self.verbose:
                print('  Checking if we can reload the weights')


            if self.use_reader:
                self.model.load_weights(self.reader, verbose=True)
            else:
                self.model.load_weights(self.weights_filename)

        else:
            if self.verbose:
                print('  ResetWeightsCallback: epoch %03d. Loading weights from %s' % (epoch, self.weights_filename))

            #self.model.load_pretrained_weights_from_reader(self.reader, quant=True)
            if self.use_reader:
                self.model.load_weights(self.reader, verbose=False)
            else:
                self.model.load_weights(self.weights_filename)

