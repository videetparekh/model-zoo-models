'''
The Keras-compatible loss function for the SSD model. Currently supports TensorFlow only.

Copyright (C) 2017 Pierluigi Ferrari

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

from __future__ import division
import tensorflow.compat.v1 as tf
from tensorflow.python.ops import array_ops
import keras
import ssd.ssd_utils as ssd_utl
import keras.backend as K

class SSDLoss(object):
    '''
    The SSD loss, see https://arxiv.org/abs/1512.02325.
    '''

    def __init__(self, ssd_config,
                 neg_pos_ratio=None,
                 n_neg_min=None,
                 alpha=None,
                 soft_labels=False,
                 use_focal_loss=False,
                 gamma=2.0,
                 fl_alpha=0.5,  # alpha for focal loss
                 label_smoothing=0,
    ):
        '''
        Arguments:
            neg_pos_ratio (int, optional): The maximum ratio of negative (i.e. background)
                to positive ground truth boxes to include in the loss computation.
                There are no actual background ground truth boxes of course, but `y_true`
                contains anchor boxes labeled with the background class. Since
                the number of background boxes in `y_true` will usually exceed
                the number of positive boxes by far, it is necessary to balance
                their influence on the loss. Defaults to 3 following the paper.
                ##if neg_pos_ratio == -1, ALL boxes are used in the loss
                ##(for distillation loss, to match HP and LP predictions)
            n_neg_min (int, optional): The minimum number of negative ground truth boxes to
                enter the loss computation *per batch*. This argument can be used to make
                sure that the model learns from a minimum number of negatives in batches
                in which there are very few, or even none at all, positive ground truth
                boxes. It defaults to 0 and if used, it should be set to a value that
                stands in reasonable proportion to the batch size used for training.
            alpha (float, optional): A factor to weight the localization loss in the
                computation of the total loss. Defaults to 1.0 following the paper.
            soft_labels (bool): True if using distillation loss, the 'y_true'
                is just the hp output, so to determine which are the positive
                vs negative samples, a argmax has to be applied.

        '''
        assert ssd_config is not None
        ssd_config = ssd_config.copy()
        if neg_pos_ratio is None:
            neg_pos_ratio = ssd_config.loss_neg_pos_ratio
        if n_neg_min is None:
            n_neg_min = ssd_config.loss_n_neg_min

        if gamma is None:
            gamma = 2.0
        if alpha is None:
            alpha = ssd_config.loss_alpha

        self.neg_pos_ratio = int(neg_pos_ratio)
        self.n_neg_min = int(n_neg_min)
        self.alpha = float(alpha)
        self.use_focal_loss = use_focal_loss
        self.gamma = gamma
        self.fl_alpha = fl_alpha

        if not bool(label_smoothing): # handle 'None' or 'False'
            label_smoothing = 0
        self.label_smoothing = label_smoothing
        self.soft_labels = soft_labels

        self.ssd_config = ssd_config


    #def __str__(self):
    #    return 'SSDLoss'

    def smooth_L1_loss(self, y_true, y_pred):
        '''
        Compute smooth L1 loss, see references.

        Arguments:
            y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
                In this context, the expected tensor has shape `(batch_size, #boxes, 4)` and
                contains the ground truth bounding box coordinates, where the last dimension
                contains `(xmin, xmax, ymin, ymax)`.
            y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
                the predicted data, in this context the predicted bounding box coordinates.

        Returns:
            The smooth L1 loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
            of shape (batch, n_boxes_total).

        References:
            https://arxiv.org/abs/1504.08083
        '''
        absolute_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred)**2
        l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
        return tf.cast(tf.reduce_sum(l1_loss, axis=-1), tf.float32)

    def focal_loss(self, y_true, y_pred):
        # y_pred (tensor of shape (batch size, num_boxes, num_classes)
        eps = K.epsilon()
        y_pred = K.clip(y_pred, eps, 1. - eps)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        loss = -K.pow(1. - pt, self.gamma) * K.log(pt)
        #loss = alpha * loss
        return tf.reduce_sum(loss, axis=-1)

    def log_loss(self, y_true, y_pred, use_focal_loss=False):
        '''
        Compute the softmax log loss.

        Arguments:
            y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
                In this context, the expected tensor has shape (batch_size, #boxes, #classes)
                and contains the ground truth bounding box categories.
            y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
                the predicted data, in this context the predicted bounding box categories.

        Returns:
            The softmax log loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
            of shape (batch, n_boxes_total).
        '''
        # Make sure that `y_pred` doesn't contain any zeros (which would break the log function)

        # Compute the log loss
        #y_pred = tf.maximum(y_pred, 1e-15)
        #log_loss = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
        config = self.ssd_config
        # if have not done softmax (or sigmoid) activation within the network, then the outputs are logits

        from_logits = not config.class_activation_in_network
        convert_logits_now = False or use_focal_loss
        #print('convert_logits_now', convert_logits_now)


        if from_logits and convert_logits_now:
            if config.class_activation == 'softmax':
                y_pred = K.softmax(y_pred)
            elif config.class_activation == 'sigmoid':
                y_pred = tf.nn.sigmoid(y_pred)
            from_logits = False

        if use_focal_loss:
            assert config.class_activation == 'softmax' and convert_logits_now
            log_loss = self.focal_loss(y_true, y_pred)
            return log_loss

        if config.class_activation == 'softmax':


            log_loss = ssd_utl.categorical_crossentropy(
                y_true, y_pred,
                from_logits=from_logits,
                label_smoothing=self.label_smoothing)

        elif config.class_activation == 'sigmoid':
            # if have not applied the sigmoid score conversion ('class_activation_in_network') within the network, do it now
            if self.soft_labels: # y_true is really the hp outputs: need to do normalization
                y_true = tf.nn.sigmoid(y_true)
                y_true = y_true /  tf.reduce_sum(y_true, axis=-1, keepdims=True)

            #log_loss = tf.reduce_sum( tf.nn.sigmoid_cross_entropy_with_logits(
            #    labels=y_true, logits=y_pred), axis=-1 )
            log_loss = ssd_utl.binary_crossentropy(y_true, y_pred, from_logits=from_logits)
        else:
            raise ValueError('Unrecognized score converter')

        '''
        if self.use_focal_loss:
            if not from_logits or convert_logits_now:
                y_pred_probs = y_pred
            else:
                y_pred_probs = K.softmax(y_pred)

            modulating_factor = tf.pow(tf.subtract(1., y_pred_probs), self.gamma)
            # use y_true to filter out y_pred for ground-truth class [reduce_sum would work too]
            focal_loss_weight = tf.reduce_max(y_true * modulating_factor, axis=-1)

            #alpha_weight = self.fl_alpha * positives + (1 - self.fl_alpha) * negatives

            log_loss *= focal_loss_weight #  = tf.multiply(alpha_weight, tf.multiply(focal_loss_weight, classification_loss))
            #focal_class_loss = tf.reduce_sum(focal_class_loss, axis=-1)  # Tensor of shape (batch_size,)
        '''


        log_loss = tf.cast(log_loss, tf.float32)
        return log_loss


    def get_losses_and_metrics(self):
        loss = self.compute_loss

        #metrics = [self.acc_pos, self.acc_all, self.dist, self.npos, self.nneg, self.prec, self.rcl]
        #metrics = [self.acc_pos, self.acc_all, self.dist, self.prec, self.rcl, self.f1]
        #metrics = [self.alt_loss, self.acc_pos, self.dist,  self.prec, self.rcl, self.f1]  # self.loc_loss,
        metrics = [self.alt_loss, self.acc_pos, self.acc_all, self.loc_loss, self.dist]  # self.loc_loss,

        return loss, metrics

    def compute_loss(self, y_true, y_pred):
        if self.use_focal_loss:
            # If the primary loss is the focal loss
            return self.compute_loss_or_metric(y_true, y_pred, to_calc='focal_loss')
        else:
            # If the primary loss is the standard ssd loss
            return self.compute_loss_or_metric(y_true, y_pred, to_calc='ssd_loss')

    def alt_loss(self, y_true, y_pred):

        if self.use_focal_loss:
            # If the primary loss is the focal loss, the alternate is the standard ssd loss
            return self.compute_loss_or_metric(y_true, y_pred, to_calc='ssd_loss')
        else:
            # If the primary loss is the standard ssd loss, the alternate is the focal loss
            return self.compute_loss_or_metric(y_true, y_pred, to_calc='focal_loss')

    def acc_pos(self, y_true, y_pred):
        return self.compute_loss_or_metric(y_true, y_pred, to_calc='acc_pos')

    def acc_all(self, y_true, y_pred):
        return self.compute_loss_or_metric(y_true, y_pred, to_calc='acc_all')

    def prec(self, y_true, y_pred):
        return self.compute_loss_or_metric(y_true, y_pred, to_calc='precision')

    def rcl(self, y_true, y_pred):
        return self.compute_loss_or_metric(y_true, y_pred, to_calc='recall')

    def f1(self, y_true, y_pred):
        return self.compute_loss_or_metric(y_true, y_pred, to_calc='f1')

    def npos(self, y_true, y_pred):
        return self.compute_loss_or_metric(y_true, y_pred, to_calc='num_pos')

    def nneg(self, y_true, y_pred):
        return self.compute_loss_or_metric(y_true, y_pred, to_calc='num_neg')

    def dist(self, y_true, y_pred):
        return self.compute_loss_or_metric(y_true, y_pred, to_calc='dist')

    def loc_loss(self, y_true, y_pred):
        return self.compute_loss_or_metric(y_true, y_pred, to_calc='loc_loss')

    def mAP(self, y_true, y_pred):
        return self.compute_mAP(y_true, y_pred)

    def compute_loss_or_metric(self, y_true, y_pred, to_calc='loss'):

        '''
        Compute the loss of the SSD model prediction against the ground truth.

        Arguments:
            y_true (array): A Numpy array of shape `(batch_size, #boxes, #classes + 12)`,
                where `#boxes` is the total number of boxes that the model predicts
                per image. Be careful to make sure that the index of each given
                box in `y_true` is the same as the index for the corresponding
                box in `y_pred`. The last axis must have length `#classes + 12` and contain
                `[classes one-hot encoded, 4 ground truth box coordinate offsets, 8 arbitrary entries]`
                in this order, including the background class. The last eight entries of the
                last axis are not used by this function and therefore their contents are
                irrelevant, they only exist so that `y_true` has the same shape as `y_pred`,
                where the last four entries of the last axis contain the anchor box
                coordinates, which are needed during inference. Important: Boxes that
                you want the cost function to ignore need to have a one-hot
                class vector of all zeros.
            y_pred (Keras tensor): The model prediction. The shape is identical
                to that of `y_true`, i.e. `(batch_size, #boxes, #classes + 12)`.
                The last axis must contain entries in the format
                `[classes one-hot encoded, 4 predicted box coordinate offsets, 8 arbitrary entries]`.

            to_calc (string) can be 'loss', 'acc_pos', 'acc_all', 'dist'

        Returns:
            A scalar, the total multitask loss for classification and localization.
        '''
        if not isinstance(self.neg_pos_ratio, tf.Tensor):
            self.neg_pos_ratio = tf.constant(self.neg_pos_ratio)

        if not isinstance(self.n_neg_min, tf.Tensor):
            self.n_neg_min = tf.constant(self.n_neg_min)

        if not isinstance(self.alpha, tf.Tensor):
            self.alpha = tf.constant(self.alpha)

        assert to_calc in ['focal_loss', 'ssd_loss',
                           'loc_loss', 'acc_pos', 'acc_all', 'f1', 'dist', 'num_pos', 'num_neg',
                           'precision', 'recall']

        if to_calc == 'loss':
            if self.use_focal_loss:
                to_calc = 'focal_loss'
            else:
                to_calc = 'ssd_loss'

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        #if self.ssd_config.add_variances_to_anchors:  # keras version
        #    box_size = 33
        #else:  # tensorflow version
        #    box_size = 99

        # y_true = tf.reshape(y_true, [-1, 1917, box_size])
        # y_pred = tf.reshape(y_pred, [-1, 1917, box_size])

        batch_size = tf.shape(y_pred)[0] # Output dtype: tf.int32
        n_boxes = tf.shape(y_pred)[1] # Output dtype: tf.int32, note that `n_boxes` in this context denotes the total number of boxes per image, not the number of boxes per cell

        config = self.ssd_config
        n_classes_w_bkg = 1 + config.n_classes
        # n_variances = len(ssd_config.variances) if ssd_config.add_variances_to_anchors else 0
        # n_box_coords = 4
        # n_anchor_coords = 4
        # expected_size = self.n_classes + n_variances + n_box_coords + n_anchor_coords
        eps = keras.backend.epsilon()

        class_slice = slice(0,  n_classes_w_bkg)
        class_slice_no_bkg = slice(1, n_classes_w_bkg)
        coords_slice = slice(n_classes_w_bkg, n_classes_w_bkg + 4)

        # 1: Compute the losses for class and box predictions for every box
        y_true_classes = y_true[:, :, class_slice]
        y_pred_classes = y_pred[:, :, class_slice]

        y_true_coords = y_true[:, :, coords_slice]
        y_pred_coords = y_pred[:, :, coords_slice]

        if to_calc in ['precision', 'recall', 'f1']:
            #
            # Take into account the fact that some boxes are neither true-positives not true-negatives:
            # boxes with intermediate IOU are just ignored. (all class_ids are false)
            class_used = tf.reduce_sum(y_true_classes, axis=-1) > 0
            true_detection = tf.logical_and( class_used, tf.argmax(y_true_classes, axis=-1) > 0)
            pred_detection = tf.logical_and( class_used, tf.argmax(y_pred_classes, axis=-1) > 0)

            true_pos = tf.reduce_sum( tf.to_float(tf.logical_and(
                pred_detection, true_detection)), axis=-1)
            false_pos = tf.reduce_sum( tf.to_float(tf.logical_and(
                pred_detection, tf.logical_not(true_detection), )), axis=-1)
            false_neg = tf.reduce_sum( tf.to_float(tf.logical_and(
                tf.logical_not(pred_detection), true_detection )), axis=-1)

            # Precision = TP / (TP + FP)
            precision = true_pos / (true_pos + false_pos + eps)

            # Recall = TP / (TP + FN)
            recall = true_pos / (true_pos + false_neg + eps)

            # F1 = 2 * P * R / (P + R)
            f1_score = 2 * precision * recall / (precision + recall + eps)

            if to_calc == 'precision':
                return precision * 100

            elif to_calc == 'recall':
                return recall * 100

            elif to_calc == 'f1':
                return f1_score * 100




        # 2: Compute the classification losses for the positive and negative targets



        # Create masks for the positive and negative ground truth classes
        if self.neg_pos_ratio == -1:  # use ALL boxes for loss function
            positives = tf.ones(y_true[..., 0].shape)
            negatives = tf.zeros(y_true[..., 0].shape)

        else:
            if self.soft_labels:
                y_true_labels = tf.argmax(y_true[:, :, class_slice], axis=-1)
                positives = tf.to_float( tf.equal(y_true_labels, 0) )
                negatives = tf.to_float( tf.greater(y_true_labels, 0) )
            else:
                negatives = y_true[:, :, 0] # Tensor of shape (batch_size, n_boxes)
                positives = tf.to_float(tf.reduce_max(y_true[:, :, class_slice_no_bkg], axis=-1)) # Tensor of shape (batch_size, n_boxes)

        # Count the number of positive boxes (classes 1 to n) in y_true across the whole batch
        n_positive = tf.reduce_sum(positives)

        # We set use_focal_loss=True if (a) we are calculating focal loss as the 'primary' loss
        #  (b) we are calculating focal loss as the 'alternate' loss (as a metric for debugging)
        #

        localization_loss = self.smooth_L1_loss(y_true_coords, y_pred_coords)  # Output shape: (batch_size, n_boxes)

        classification_loss_focal = self.log_loss(y_true_classes, y_pred_classes, use_focal_loss=True)  # Output shape: (batch_size, n_boxes)
        classification_loss_ssd   = self.log_loss(y_true_classes, y_pred_classes, use_focal_loss=False)  # Output shape: (batch_size, n_boxes)

        if self.use_focal_loss and (to_calc != 'ssd_loss'):
            # Avoid having to calculate classification/localization loss unless we actually need it.
            # (when calculating some of the metrics, we need the positives/negatives, but not the actual cross-entropy loss)
            classification_loss_for_neg_keep = classification_loss_focal
            # use all negative examples
            negatives_keep = negatives
            n_neg_losses = None
        else:
            # calculate the standard ssd cross-entropy loss
            # (we always need to calculate the loss, even for metrics, so that we know which are the negatives we will keep)
            # Compute the classification loss for the negative default boxes (if there are any)
            classification_loss_for_neg_keep = classification_loss_ssd

            # First, compute the classification loss for all negative boxes
            neg_class_loss_all = classification_loss_for_neg_keep * negatives # Tensor of shape (batch_size, n_boxes)
            n_neg_losses = tf.count_nonzero(neg_class_loss_all, dtype=tf.int32) # The number of non-zero loss entries in `neg_class_loss_all`
            # What's the point of `n_neg_losses`? For the next step, which will be to compute which negative boxes enter the classification
            # loss, we don't just want to know how many negative ground truth boxes there are, but for how many of those there actually is
            # a positive (i.e. non-zero) loss. This is necessary because `tf.nn.top-k()` in the function below will pick the top k boxes with
            # the highest losses no matter what, even if it receives a vector where all losses are zero. In the unlikely event that all negative
            # classification losses ARE actually zero though, this behavior might lead to `tf.nn.top-k()` returning the indices of positive
            # boxes, leading to an incorrect negative classification loss computation, and hence an incorrect overall loss computation.
            # We therefore need to make sure that `n_negative_keep`, which assumes the role of the `k` argument in `tf.nn.top-k()`,
            # is at most the number of negative boxes for which there is a positive classification loss.

            # Compute the number of negative examples we want to account for in the loss
            # We'll keep at most `self.neg_pos_ratio` times the number of positives in `y_true`, but at least `self.n_neg_min` (unless `n_neg_loses` is smaller)
            n_negative_keep = tf.minimum(tf.maximum(self.neg_pos_ratio * tf.to_int32(n_positive), self.n_neg_min), n_neg_losses)


            # Now we'll identify the top-k (where k == `n_negative_keep`) boxes with the highest confidence loss that
            # belong to the background class in the ground truth data. Note that this doesn't necessarily mean that the model
            # predicted the wrong class for those boxes, it just means that the loss for those boxes is the highest.

            # To do this, we reshape `neg_class_loss_all` to 1D...
            neg_class_loss_all_1D = tf.reshape(neg_class_loss_all, [-1])  # Tensor of shape (batch_size * n_boxes,)
            # ...and then we get the indices for the `n_negative_keep` boxes with the highest loss out of those...
            values, indices = tf.nn.top_k(neg_class_loss_all_1D, n_negative_keep, False)  # We don't need sorting
            # ...and with these indices we'll create a mask...
            negatives_keep = tf.scatter_nd(tf.expand_dims(indices, axis=1), updates=tf.ones_like(indices, dtype=tf.int32),
                                           shape=tf.shape(neg_class_loss_all_1D))  # Tensor of shape (batch_size * n_boxes,)
            negatives_keep = tf.to_float(
                tf.reshape(negatives_keep, [batch_size, n_boxes]))  # Tensor of shape (batch_size, n_boxes)


        if to_calc in ['focal_loss', 'ssd_loss', 'loc_loss']:

            # 3: Compute the localization loss for the positive targets
            #    We don't penalize localization loss for negative predicted boxes (obviously: there are no ground truth boxes they would correspond to)

            loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1) # Tensor of shape (batch_size,)
            if to_calc == 'loc_loss':
                return loc_loss

            if to_calc == 'focal_loss':
                classification_loss_use = classification_loss_focal
                if self.fl_alpha is not None and self.fl_alpha != 0.5:
                    alpha_weight = (self.fl_alpha * positives + (1 - self.fl_alpha) * negatives) * 2.0  # x2 to reduce to identity when alpha = 0.5
                    classification_loss_use *= alpha_weight

                class_loss = tf.reduce_sum(classification_loss_use, axis=-1)  # Tensor of shape (batch_size,)

            else:
                assert to_calc == 'ssd_loss'
                # Now mask all negative boxes and sum up the losses for the positive boxes PER batch item
                # (Keras loss functions must output one scalar loss value PER batch item, rather than just
                # one scalar for the entire batch, that's why we're not summing across all axes)
                classification_loss_use = classification_loss_ssd
                pos_class_loss = tf.reduce_sum(classification_loss_use * positives, axis=-1)  # Tensor of shape (batch_size,)

                # In the unlikely case when either (1) there are no negative ground truth boxes at all
                # or (2) the classification loss for all negative boxes is zero, return zero as the `neg_class_loss`
                def f1():
                    return tf.zeros([batch_size])

                # Otherwise compute the negative loss
                def f2():
                    # ...and use it to keep only those boxes and mask all other classification losses
                    neg_class_loss = tf.reduce_sum(classification_loss_use * negatives_keep,
                                                   axis=-1)  # Tensor of shape (batch_size,)
                    return neg_class_loss

                neg_class_loss = tf.cond(tf.equal(n_neg_losses, tf.constant(0)), f1, f2)

                class_loss = pos_class_loss + neg_class_loss  # Tensor of shape (batch_size,)


            # 4: Compute the total loss
            #self.class_loss = class_loss / tf.maximum(1.0, n_positive)
            #self.loc_loss = (self.alpha * loc_loss) / tf.maximum(1.0, n_positive)

            total_loss = (class_loss + self.alpha * loc_loss) / tf.maximum(1.0, n_positive) # In case `n_positive == 0`
            # Keras has the annoying habit of dividing the loss by the batch size, which sucks in our case
            # because the relevant criterion to average our loss over is the number of positive boxes in the batch
            # (by which we're dividing in the line above), not the batch size. So in order to revert Keras' averaging
            # over the batch size, we'll have to multiply by it.
            total_loss *= tf.to_float(batch_size)
            #print "Classification loss: ", self.class_loss
            #print "Localization loss: ", self.loc_loss
            return total_loss


        elif to_calc in ['acc_pos', 'acc_all']:
            classfication_acc_all = keras.metrics.categorical_accuracy(y_true_classes, y_pred_classes)*100
            if to_calc == 'acc_pos':
                bool_use = positives
                # classification metric of positive classes only:
            elif to_calc == 'acc_all':
                # classification metric of all trained boxes (positive + negative):
                # when primary loss == focal loss, this includes ALL boxes
                bool_use = positives + negatives_keep
            else:
                raise ValueError('')

            classfication_acc_use = tf.reduce_sum( classfication_acc_all * bool_use, axis=1) / (tf.reduce_sum(bool_use, axis=1) + eps)
            return classfication_acc_use


        elif to_calc == 'num_pos':
            return tf.reduce_sum(positives, axis=-1)

        elif to_calc == 'num_neg':
            return tf.reduce_sum(negatives_keep, axis=-1)

        elif to_calc == 'dist':
            # Distance only matters for the positive examples.
            y_true_decoded = ssd_utl.decode_y_coordinates(y_true, config, final_coords='corners', remove_anchors=True)
            y_pred_decoded = ssd_utl.decode_y_coordinates(y_pred, config, final_coords='corners', remove_anchors=True)

            y_true_coords_decoded = y_true_decoded[:, :, coords_slice]
            y_pred_coords_decoded = y_pred_decoded[:, :, coords_slice]
            if config.normalize_coords:
                width, height = config.img_width, config.img_height
                size_vector = tf.to_float( tf.stack( (width, height, width, height)) )
                y_true_coords_decoded = y_true_coords_decoded * size_vector
                y_pred_coords_decoded = y_pred_coords_decoded * size_vector

            dist_acc_all = tf.reduce_mean( tf.abs( y_true_coords_decoded - y_pred_coords_decoded ), axis=2)
            dist_pos     = tf.reduce_sum( dist_acc_all * positives, axis=1)  / (tf.reduce_sum(positives, axis=1) + eps)
            return dist_pos

        else:
            raise ValueError('Unhandled metric/loss : %s ' % to_calc)



    def compute_mAP(self, y_true, y_pred):
        pass





class FocalLoss(object):
    '''
    The SSD loss, see https://arxiv.org/abs/1512.02325.
    '''

    def __init__(self,
                 neg_pos_ratio=3,
                 n_neg_min=0,
                 alpha=1.0):
        '''
        Arguments:
            neg_pos_ratio (int, optional): The maximum ratio of negative (i.e. background)
                to positive ground truth boxes to include in the loss computation.
                There are no actual background ground truth boxes of course, but `y_true`
                contains anchor boxes labeled with the background class. Since
                the number of background boxes in `y_true` will usually exceed
                the number of positive boxes by far, it is necessary to balance
                their influence on the loss. Defaults to 3 following the paper.
            n_neg_min (int, optional): The minimum number of negative ground truth boxes to
                enter the loss computation *per batch*. This argument can be used to make
                sure that the model learns from a minimum number of negatives in batches
                in which there are very few, or even none at all, positive ground truth
                boxes. It defaults to 0 and if used, it should be set to a value that
                stands in reasonable proportion to the batch size used for training.
            alpha (float, optional): A factor to weight the localization loss in the
                computation of the total loss. Defaults to 1.0 following the paper.
        '''
        self.neg_pos_ratio = neg_pos_ratio
        self.n_neg_min = n_neg_min
        self.alpha = alpha

    def smooth_L1_loss(self, y_true, y_pred):
        '''
        Compute smooth L1 loss, see references.

        Arguments:
            y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
                In this context, the expected tensor has shape `(batch_size, #boxes, 4)` and
                contains the ground truth bounding box coordinates, where the last dimension
                contains `(xmin, xmax, ymin, ymax)`.
            y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
                the predicted data, in this context the predicted bounding box coordinates.

        Returns:
            The smooth L1 loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
            of shape (batch, n_boxes_total).

        References:
            https://arxiv.org/abs/1504.08083
        '''
        absolute_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred)**2
        l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
        return tf.reduce_sum(l1_loss, axis=-1)

    def log_loss(self, y_true, y_pred, gamma=2, alpha=0.5):
        '''
        Compute the softmax log loss.

        Arguments:
            y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
                In this context, the expected tensor has shape (batch_size, #boxes, #classes)
                and contains the ground truth bounding box categories.
            y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
                the predicted data, in this context the predicted bounding box categories.

        Returns:
            The softmax log loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
            of shape (batch, n_boxes_total).
        '''
        # Make sure that `y_pred` doesn't contain any zeros (which would break the log function)
        #sigmoid_p = tf.nn.sigmoid(y_pred)
        # zeros = array_ops.zeros_like(y_pred, dtype=y_pred.dtype)
    
        # # For poitive prediction, only need consider front part loss, back part is 0;
        # # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
        # pos_p_sub = array_ops.where(y_true > zeros, y_true - y_pred, zeros)
    
        # # For negative prediction, only need consider back part loss, front part is 0;
        # # target_tensor > zeros <=> z=1, so negative coefficient = 0.
        # neg_p_sub = array_ops.where(y_true > zeros, zeros, y_pred)
        # per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(y_pred, 1e-8, 1.0)) \
        #                   - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - y_pred, 1e-8, 1.0))
        # return tf.reduce_sum(per_entry_cross_ent)
        y_pred = tf.maximum(y_pred, 1e-15)
        log_y_pred = tf.log(y_pred)
        focal_scale = tf.multiply(tf.pow(tf.subtract(1.0, y_pred), gamma), alpha)
        focal_loss = tf.multiply(y_true, tf.multiply(focal_scale, log_y_pred))
        return -tf.reduce_sum(focal_loss, axis=-1)

    def compute_loss(self, y_true, y_pred):
        '''
        Compute the loss of the SSD model prediction against the ground truth.

        Arguments:
            y_true (array): A Numpy array of shape `(batch_size, #boxes, #classes + 12)`,
                where `#boxes` is the total number of boxes that the model predicts
                per image. Be careful to make sure that the index of each given
                box in `y_true` is the same as the index for the corresponding
                box in `y_pred`. The last axis must have length `#classes + 12` and contain
                `[classes one-hot encoded, 4 ground truth box coordinate offsets, 8 arbitrary entries]`
                in this order, including the background class. The last eight entries of the
                last axis are not used by this function and therefore their contents are
                irrelevant, they only exist so that `y_true` has the same shape as `y_pred`,
                where the last four entries of the last axis contain the anchor box
                coordinates, which are needed during inference. Important: Boxes that
                you want the cost function to ignore need to have a one-hot
                class vector of all zeros.
            y_pred (Keras tensor): The model prediction. The shape is identical
                to that of `y_true`, i.e. `(batch_size, #boxes, #classes + 12)`.
                The last axis must contain entries in the format
                `[classes one-hot encoded, 4 predicted box coordinate offsets, 8 arbitrary entries]`.

        Returns:
            A scalar, the total multitask loss for classification and localization.
        '''
        self.neg_pos_ratio = tf.constant(self.neg_pos_ratio)
        self.n_neg_min = tf.constant(self.n_neg_min)
        self.alpha = tf.constant(self.alpha)

        batch_size = tf.shape(y_pred)[0] # Output dtype: tf.int32
        n_boxes = tf.shape(y_pred)[1] # Output dtype: tf.int32, note that `n_boxes` in this context denotes the total number of boxes per image, not the number of boxes per cell

        # 1: Compute the losses for class and box predictions for every box

        classification_loss = tf.to_float(self.log_loss(y_true[:,:,:-12], y_pred[:,:,:-12])) # Output shape: (batch_size, n_boxes)
        localization_loss = tf.to_float(self.smooth_L1_loss(y_true[:,:,-12:-8], y_pred[:,:,-12:-8])) # Output shape: (batch_size, n_boxes)

        # 2: Compute the classification losses for the positive and negative targets

        # Create masks for the positive and negative ground truth classes
        negatives = y_true[:,:,0] # Tensor of shape (batch_size, n_boxes)
        positives = tf.to_float(tf.reduce_max(y_true[:,:,1:-12], axis=-1)) # Tensor of shape (batch_size, n_boxes)

        # Count the number of positive boxes (classes 1 to n) in y_true across the whole batch
        n_positive = tf.reduce_sum(positives)

        # Now mask all negative boxes and sum up the losses for the positive boxes PER batch item
        # (Keras loss functions must output one scalar loss value PER batch item, rather than just
        # one scalar for the entire batch, that's why we're not summing across all axes)
        pos_class_loss = tf.reduce_sum(classification_loss * positives, axis=-1) # Tensor of shape (batch_size,)

        # Compute the classification loss for the negative default boxes (if there are any)

        # First, compute the classification loss for all negative boxes
        neg_class_loss_all = classification_loss * negatives # Tensor of shape (batch_size, n_boxes)
        n_neg_losses = tf.count_nonzero(neg_class_loss_all, dtype=tf.int32) # The number of non-zero loss entries in `neg_class_loss_all`
        # What's the point of `n_neg_losses`? For the next step, which will be to compute which negative boxes enter the classification
        # loss, we don't just want to know how many negative ground truth boxes there are, but for how many of those there actually is
        # a positive (i.e. non-zero) loss. This is necessary because `tf.nn.top-k()` in the function below will pick the top k boxes with
        # the highest losses no matter what, even if it receives a vector where all losses are zero. In the unlikely event that all negative
        # classification losses ARE actually zero though, this behavior might lead to `tf.nn.top-k()` returning the indices of positive
        # boxes, leading to an incorrect negative classification loss computation, and hence an incorrect overall loss computation.
        # We therefore need to make sure that `n_negative_keep`, which assumes the role of the `k` argument in `tf.nn.top-k()`,
        # is at most the number of negative boxes for which there is a positive classification loss.

        # Compute the number of negative examples we want to account for in the loss
        # We'll keep at most `self.neg_pos_ratio` times the number of positives in `y_true`, but at least `self.n_neg_min` (unless `n_neg_loses` is smaller)
        n_negative_keep = tf.minimum(tf.maximum(self.neg_pos_ratio * tf.to_int32(n_positive), self.n_neg_min), n_neg_losses)

        # In the unlikely case when either (1) there are no negative ground truth boxes at all
        # or (2) the classification loss for all negative boxes is zero, return zero as the `neg_class_loss`
        def f1():
            return tf.zeros([batch_size])
        # Otherwise compute the negative loss
        def f2():
            # Now we'll identify the top-k (where k == `n_negative_keep`) boxes with the highest confidence loss that
            # belong to the background class in the ground truth data. Note that this doesn't necessarily mean that the model
            # predicted the wrong class for those boxes, it just means that the loss for those boxes is the highest.

            # To do this, we reshape `neg_class_loss_all` to 1D...
            neg_class_loss_all_1D = tf.reshape(neg_class_loss_all, [-1]) # Tensor of shape (batch_size * n_boxes,)
            # ...and then we get the indices for the `n_negative_keep` boxes with the highest loss out of those...
            values, indices = tf.nn.top_k(neg_class_loss_all_1D, n_negative_keep, False) # We don't need sorting
            # ...and with these indices we'll create a mask...
            negatives_keep = tf.scatter_nd(tf.expand_dims(indices, axis=1), updates=tf.ones_like(indices, dtype=tf.int32), shape=tf.shape(neg_class_loss_all_1D)) # Tensor of shape (batch_size * n_boxes,)
            negatives_keep = tf.to_float(tf.reshape(negatives_keep, [batch_size, n_boxes])) # Tensor of shape (batch_size, n_boxes)
            # ...and use it to keep only those boxes and mask all other classification losses
            neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep, axis=-1) # Tensor of shape (batch_size,)
            return neg_class_loss

        neg_class_loss = tf.cond(tf.equal(n_neg_losses, tf.constant(0)), f1, f2)

        class_loss = pos_class_loss + neg_class_loss # Tensor of shape (batch_size,)

        # 3: Compute the localization loss for the positive targets
        #    We don't penalize localization loss for negative predicted boxes (obviously: there are no ground truth boxes they would correspond to)

        loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1) # Tensor of shape (batch_size,)

        # 4: Compute the total loss

        total_loss = (class_loss + self.alpha * loc_loss) / tf.maximum(1.0, n_positive) # In case `n_positive == 0`
        # Keras has the annoying habit of dividing the loss by the batch size, which sucks in our case
        # because the relevant criterion to average our loss over is the number of positive boxes in the batch
        # (by which we're dividing in the line above), not the batch size. So in order to revert Keras' averaging
        # over the batch size, we'll have to multiply by it.
        total_loss *= tf.to_float(batch_size)

        return total_loss

class weightedSSDLoss(SSDLoss):
    def __init__(self,
                 neg_pos_ratio=3,
                 n_neg_min=0,
                 alpha=1.0,
                 weights=None):

        super(weightedSSDLoss, self).__init__(neg_pos_ratio, 
                                              n_neg_min,
                                              alpha)
        self.weights = weights

    def log_loss(self, y_true, y_pred):
        
        weighted = tf.multiply(y_true, self.weights)
        y_pred = tf.maximum(y_pred, 1e-15)
        # Compute the log loss
        xent = tf.multiply(y_true, tf.log(y_pred))
        log_loss = -tf.reduce_sum(weighted * xent, axis=-1)
        return log_loss


class weightedFocalLoss(FocalLoss):
    def __init__(self,
                 neg_pos_ratio=3,
                 n_neg_min=0,
                 alpha=1.0,
                 weights=None):

        super(weightedFocalLoss, self).__init__(neg_pos_ratio, 
                                              n_neg_min,
                                              alpha)
        self.weights = weights

    def log_loss(self, y_true, y_pred, gamma=2, alpha=0.5):
        
        weighted = tf.multiply(y_true, self.weights)
        y_pred = tf.maximum(y_pred, 1e-15)
        log_y_pred = tf.log(y_pred)
        focal_scale = tf.multiply(tf.pow(tf.subtract(1.0, y_pred), gamma), alpha)
        focal_loss = tf.multiply(weighted, tf.multiply(focal_scale, log_y_pred))
        return -tf.reduce_sum(focal_loss, axis=-1)



if __name__ == "__main__":

    import hickle as hkl
    import networks


    S = hkl.load('F:/SRI/bitnet/ssd_pred_true.hkl')
    y_pred = S['y_pred']
    y_true = S['y_true']
    S_save = dict(y_pred=y_pred, y_true=y_true)

    net_type = networks.net_types.SSDNetType(mobilenet_version=2, style='tf')
    ssd_config = ssd_utl.SSDConfig(net_type=net_type)

    ssd_loss = SSDLoss(ssd_config=ssd_config)

    loss2 = ssd_loss.compute_loss(y_true, y_pred)

    a = 1
