from tensorflow.keras import backend as K
import tensorflow as tf

# had to replace sometimes K by tf, because else: TypeError: An op outside of the function building code is being passed
#     a "Graph" tensor. It is possible to have Graph tensors
#     leak out of the function building context by including a
#     tf.init_scope in your function building code.
# Reason was: So the main issue here is that custom loss function is returning a Symbolic KerasTensor and not a Tensor.
#     And this is happening because inputs to the custom loss function are in Symbolic KerasTensor form.
#     ref: https://github.com/tensorflow/tensorflow/issues/43650

# Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
# -> the score is computed for each class separately and then summed
# alpha=beta=0.5 : dice coefficient
# alpha=beta=1   : tanimoto coefficient (also known as jaccard)
# alpha+beta=1   : produces set of F*-scores
def tversky_loss(y_true, y_pred, alpha = 0.3, beta=0.7, epsilon = 1e-3 ):

    # Assign Alpha as A Member Variable
    tversky_loss.alpha = alpha
    tversky_loss.beta = beta
    tversky_loss.epislon = epsilon

    ones = tf.ones(tf.shape(y_true))
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class i
    g0 = y_true
    g1 = ones - y_true

    num = K.sum(p0 * g0, (0, 1, 2, 3))
    den = num + alpha * K.sum(p0 * g1, (0, 1, 2, 3)) + beta * K.sum(p1 * g0, (0, 1, 2, 3)) + epsilon

    T = K.sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]

    Ncl = tf.cast(tf.shape(y_true)[-1], 'float32')
    return Ncl - T

# Assign a name to the function
tversky_loss.name = 'tversky_loss'

def focal_tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, gamma=2.0, epsilon=1e-3):
    """
    Focal Tversky loss for multi-class segmentation.
    
    Args:
        y_true: Tensor of shape [batch_size, height, width, depth, num_classes]
                representing one-hot encoded ground truth labels.
        y_pred: Tensor of shape [batch_size, height, width, depth, num_classes]
                representing the predicted probabilities for each class.
        alpha: Weight of false positives.
        beta: Weight of false negatives.
        gamma: Focusing parameter.
        epsilon: Small value to avoid division by zero.
                
    Returns:
        loss: Focal Tversky loss.
    """

    # Assign Alpha as A Member Variable
    focal_tversky_loss.alpha = alpha
    focal_tversky_loss.beta = beta
    focal_tversky_loss.gamma = gamma    
    focal_tversky_loss.epislon = epsilon

    # Ensure y_true and y_pred are float32 tensors
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    
    # Clip predictions to prevent log(0) errors
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    
    # Calculate true positives, false negatives, and false positives
    true_pos = K.sum(y_true * y_pred, axis=[1, 2, 3, 4])
    false_neg = K.sum(y_true * (1 - y_pred), axis=[1, 2, 3, 4])
    false_pos = K.sum((1 - y_true) * y_pred, axis=[1, 2, 3, 4])
    
    # Calculate the Tversky index
    tversky_index = (true_pos + epsilon) / (true_pos + alpha * false_neg + beta * false_pos + epsilon)
    
    # Apply the focal term
    focal_tversky_loss = K.pow((1 - tversky_index), gamma)
    
    # Return the mean loss over the batch
    return K.mean(focal_tversky_loss)

# Assign a name to the function
focal_tversky_loss.name = 'focal_tversky_loss'
