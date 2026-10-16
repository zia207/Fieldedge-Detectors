"""
This file provides a set of utility functions for deep learning-based image segmentation tasks,
primarily in TensorFlow. The code includes functionalities for file and string handling, logging,
model introspection, preprocessing, custom loss computations, metrics calculation, dataset statistics,
and system debugging.

Key Functions and Their Explanations:

- filepath_to_name: Extracts the file name without extension from a given full file path.

- LOG: Logs a message prepended with a timestamp, either to the console or a given file.

- count_params: Prints the total number of trainable parameters in a loaded TensorFlow model.

- mean_image_subtraction: Subtracts provided mean values (typically ImageNet means) channelwise from an input image tensor.

- _lovasz_grad: Computes the gradient of the Lovasz extension used for optimizing IoU-like metrics as a loss.

- _flatten_probas: Reshapes predictions and labels tensors into 2D/1D for loss computations, with optional class ignoring and layout support (BHWC/BCHW).

- _lovasz_softmax_flat: Computes the per-class Lovasz-Softmax loss from flattened probabilities and ground truth labels,
  optionally considering only classes present in the ground truth.

- lovasz_softmax: The main Lovasz-Softmax loss function. Applies softmax, preps ground truth, and computes the per-class or per-image loss.

- random_crop: Randomly crops a given image and label pair to the desired size for data augmentation. Handles both 2D and multi-channel 3D input.

- compute_global_accuracy: Calculates the overall (per-pixel) accuracy of segmentation predictions.

- compute_class_accuracies: Computes the accuracy per class, handling the case where a class is not present in the label.

- compute_mean_iou: Calculates the mean Intersection over Union (IoU) score over all unique label values in the batch.

- evaluate_segmentation: Collects several key segmentation performance metrics (global accuracy, per-class accuracy, precision, recall, F1 score, IoU) for a prediction/GT pair.

- compute_class_weights: Computes class weights for imbalanced datasets by analyzing label images. Outputs normalized class weights based on pixel frequency.

- memory: Prints the current Python process memory usage in gigabytes for debugging or profiling.

Imports:
- Uses a mix of standard Python libraries (os, sys, random, time, datetime, math, ast),
  popular scientific libraries (NumPy, SciPy, scikit-learn),
  OpenCV (cv2), TensorFlow and its contrib.slim package,
  and project-local helpers module.

This set of utilities supports tasks common in deep learning for image segmentation,
including data pipeline preparation, augmentation, evaluation, and model introspection.

"""

from __future__ import print_function, division
import os, time, cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import os, random
from scipy.misc import imread
import ast
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

import helpers


# Takes an absolute file path and returns the name of the file without th extension
def filepath_to_name(full_name):
    """
    Given an absolute file path, returns the filename without its extension.

    Example:
        filepath_to_name("/path/to/image.png")  ->  "image"
    """
    file_name = os.path.basename(full_name)
    file_name = os.path.splitext(file_name)[0]
    return file_name

# Print with time. To console or file
def LOG(X, f=None):
    """
    Prints or writes a log message X, prepended with the current timestamp.
    If f is provided, writes to the file handle f, otherwise prints to stdout.
    """
    time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if not f:
        print(time_stamp + " " + X)
    else:
        f.write(time_stamp + " " + X)


# Count total number of parameters in the model
def count_params():
    """
    Prints the number of trainable parameters in the current TensorFlow graph/model.
    """
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("This model has %d trainable parameters"% (total_parameters))

# Subtracts the mean images from ImageNet
def mean_image_subtraction(inputs, means=[123.68, 116.78, 103.94]):
    """
    Subtracts the given means from input image tensor along the channels.
    Typically used for input normalization using ImageNet mean pixel values.

    Args:
        inputs: Input tensor of shape [B, H, W, C]
        means: List of mean values, one per channel.

    Returns:
        Normalized tensor of the same shape as inputs.
    """
    inputs = tf.to_float(inputs)
    num_channels = inputs.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=inputs)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)

def _lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension with respect to sorted errors.
    Ref: "The Lovasz-Softmax loss: A tractable surrogate for the optimization
    of the intersection-over-union measure in neural networks" (Berman et al.)
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard

def _flatten_probas(probas, labels, ignore=None, order='BHWC'):
    """
    Flattens predictions and labels for batch loss computation.

    Args:
        probas: Tensor containing softmax probabilities.
        labels: Tensor of GT labels.
        ignore: (Optional) Label to ignore during the computation.
        order: Channel ordering, either 'BHWC' or 'BCHW'.

    Returns:
        probas, labels: reshaped tensors, optionally excluding ignored labels.
    """
    if order == 'BCHW':
        probas = tf.transpose(probas, (0, 2, 3, 1), name="BCHW_to_BHWC")
        order = 'BHWC'
    if order != 'BHWC':
        raise NotImplementedError('Order {} unknown'.format(order))
    C = probas.shape[3]
    probas = tf.reshape(probas, (-1, C))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return probas, labels
    valid = tf.not_equal(labels, ignore)
    vprobas = tf.boolean_mask(probas, valid, name='valid_probas')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vprobas, vlabels

def _lovasz_softmax_flat(probas, labels, only_present=True):
    """
    Computes multi-class Lovasz-Softmax loss for flattened probabilities and labels.

    Args:
        probas: [P, C] Tensor, class-wise probabilities.
        labels: [P] Tensor, ground truth labels.
        only_present: If True, averages only over classes actually present in labels.

    Returns:
        Tensor of losses.
    """
    C = probas.shape[1]
    losses = []
    present = []
    for c in range(C):
        fg = tf.cast(tf.equal(labels, c), probas.dtype) # foreground mask for class c
        if only_present:
            present.append(tf.reduce_sum(fg) > 0)
        errors = tf.abs(fg - probas[:, c])
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort_{}".format(c))
        fg_sorted = tf.gather(fg, perm)
        grad = _lovasz_grad(fg_sorted)
        losses.append(
            tf.tensordot(errors_sorted, tf.stop_gradient(grad), 1, name="loss_class_{}".format(c))
        )
    losses_tensor = tf.stack(losses)
    if only_present:
        present = tf.stack(present)
        losses_tensor = tf.boolean_mask(losses_tensor, present)
    return losses_tensor

def lovasz_softmax(probas, labels, only_present=True, per_image=False, ignore=None, order='BHWC'):
    """
    Main Lovasz-Softmax loss for multi-class segmentation.

    Args:
        probas: Network output of softmax probabilities, [B, H, W, C] or [B, C, H, W].
        labels: [B, H, W] Tensor, integer ground truth labels.
        only_present: Only consider present classes in the loss.
        per_image: If True, computes the loss per image and averages.
        ignore: Optional label to ignore.
        order: Either 'BHWC' or 'BCHW'.

    Returns:
        Lovasz softmax loss tensor.
    """
    probas = tf.nn.softmax(probas, 3)
    labels = helpers.reverse_one_hot(labels)
    if per_image:
        def treat_image(prob, lab):
            prob, lab = tf.expand_dims(prob, 0), tf.expand_dims(lab, 0)
            prob, lab = _flatten_probas(prob, lab, ignore, order)
            return _lovasz_softmax_flat(prob, lab, only_present=only_present)
        losses = tf.map_fn(treat_image, (probas, labels), dtype=tf.float32)
    else:
        losses = _lovasz_softmax_flat(*_flatten_probas(probas, labels, ignore, order), only_present=only_present)
    return losses

# Randomly crop the image to a specific size. For data augmentation
def random_crop(image, label, crop_height, crop_width):
    """
    Randomly crops both the input image and its label to (crop_height, crop_width).
    Used for data augmentation.

    Handles cases where the label or image is 2D or 3D.

    Raises:
        Exception: If the label and image shapes do not match,
                   or if requested crop size exceeds dimensions.
    """
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')
        
    if (crop_width <= image.shape[1]) and (crop_height <= image.shape[0]):
        x = random.randint(0, image.shape[1]-crop_width)
        y = random.randint(0, image.shape[0]-crop_height)
        
        if len(label.shape) == 3:
            return image[y:y+crop_height, x:x+crop_width, :], label[y:y+crop_height, x:x+crop_width, :]
        else:
            return image[y:y+crop_height, x:x+crop_width, :], label[y:y+crop_height, x:x+crop_width]
    else:
        raise Exception('Crop shape exceeds image dimensions!')

# Compute the average segmentation accuracy across all classes
def compute_global_accuracy(pred, label):
    """
    Computes the (per-pixel) global accuracy:
    Fraction of predicted pixels that are correctly labeled.

    Args:
        pred: 1D array of predicted labels.
        label: 1D array of ground truth labels.

    Returns:
        Float accuracy (between 0.0 and 1.0).
    """
    total = len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)

# Compute the class-specific segmentation accuracy
def compute_class_accuracies(pred, label, num_classes):
    """
    Computes the pixel accuracy for each class (per-class accuracy).

    Args:
        pred: 1D array of predicted labels.
        label: 1D array of GT labels.
        num_classes: Number of unique GT/pred classes.

    Returns:
        List of accuracies for each class. If the class does not appear in GT labels,
        accuracy is set to 1.0.
    """
    total = []
    for val in range(num_classes):
        total.append((label == val).sum())

    count = [0.0] * num_classes
    for i in range(len(label)):
        if pred[i] == label[i]:
            count[int(pred[i])] = count[int(pred[i])] + 1.0

    # If there are no pixels from a certain class in the GT,
    # it returns NAN because of divide by zero
    # Replace the nans with a 1.0.
    accuracies = []
    for i in range(len(total)):
        if total[i] == 0:
            accuracies.append(1.0)
        else:
            accuracies.append(count[i] / total[i])

    return accuracies

def compute_mean_iou(pred, label):
    """
    Computes the mean Intersection over Union (mIoU) between predicted and ground truth labels.

    Args:
        pred, label: 1D arrays of predicted and GT labels.

    Returns:
        Mean IoU as a float.
    """
    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels);

    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val

        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))


    mean_iou = np.mean(I / U)
    return mean_iou

def evaluate_segmentation(pred, label, num_classes, score_averaging="weighted"):
    """
    Computes summary segmentation metrics for a prediction/label pair.

    Args:
        pred: segmentation prediction output (typically 2D or 3D numpy array)
        label: ground truth mask array (same shape as pred)
        num_classes: integer, number of classes in label set
        score_averaging: string, averaging mode for sklearn metrics

    Returns:
        Tuple: (global_accuracy, class_accuracies, precision, recall, f1, IoU)
    """
    flat_pred = pred.flatten()
    flat_label = label.flatten()

    global_accuracy = compute_global_accuracy(flat_pred, flat_label)
    class_accuracies = compute_class_accuracies(flat_pred, flat_label, num_classes)

    prec = precision_score(flat_pred, flat_label, average=score_averaging)
    rec = recall_score(flat_pred, flat_label, average=score_averaging)
    f1 = f1_score(flat_pred, flat_label, average=score_averaging)

    iou = compute_mean_iou(flat_pred, flat_label)

    return global_accuracy, class_accuracies, prec, rec, f1, iou

def compute_class_weights(labels_dir, label_values):
    '''
    Computes class weights using pixel counts from a directory of label images.

    Args:
        labels_dir (str): Directory where segmentation labels are found (PNG files).
        label_values (list): List of RGB values (one per class).

    Returns:
        class_weights (numpy.array): Normalized weights for each class label (use to counteract imbalance).
    '''
    image_files = [os.path.join(labels_dir, file) for file in os.listdir(labels_dir) if file.endswith('.png')]

    num_classes = len(label_values)

    class_pixels = np.zeros(num_classes) 

    total_pixels = 0.0

    for n in range(len(image_files)):
        image = imread(image_files[n])

        for index, colour in enumerate(label_values):
            class_map = np.all(np.equal(image, colour), axis = -1)
            class_map = class_map.astype(np.float32)
            class_pixels[index] += np.sum(class_map)

        print("\rProcessing image: " + str(n) + " / " + str(len(image_files)), end="")
        sys.stdout.flush()

    total_pixels = float(np.sum(class_pixels))
    index_to_delete = np.argwhere(class_pixels==0.0)
    class_pixels = np.delete(class_pixels, index_to_delete)

    class_weights = total_pixels / class_pixels
    class_weights = class_weights / np.sum(class_weights)

    return class_weights

# Compute the memory usage, for debugging
def memory():
    """
    Prints current RAM usage (GB) of this Python process (for debugging/profiling).
    """
    import os
    import psutil
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30  # Memory use in GB
    print('Memory usage in GBs:', memoryUse)
