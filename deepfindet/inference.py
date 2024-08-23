# =============================================================================================
# DeepFinder - a deep learning approach to localize macromolecules in cryo electron tomograms
# =============================================================================================
# Copyright (C) Inria,  Emmanuel Moebel, Charles Kervrann, All Rights Reserved, 2015-2021, v1.0
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# =============================================================================================

from deepfinder.models import model_loader
from deepfinder.utils import core
import tensorflow as tf
import numpy as np
import time

# Enable mixed precision
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

class Segment(core.DeepFinder):
    def __init__(self, Ncl, model_name, path_weights, patch_size=192, gpuID = None):
        core.DeepFinder.__init__(self)

        self.Ncl = Ncl

        # Segmentation, parameters for dividing data in patches:
        self.P = patch_size  # patch length (in pixels) /!\ has to a multiple of 4 (because of 2 pooling layers), so that dim_in=dim_out
        self.pcrop = 25  # how many pixels to crop from border (net model dependent)
        self.poverlap = 55  # patch overlap (in pixels) (2*pcrop + 5)

        self.path_weights = path_weights
        self.check_attributes()

        # Initialize Empty network:
        self.net = model_loader.load_model(patch_size, Ncl, model_name, path_weights)

        # Set GPU configuration
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus and gpuID is not None:
            try:
                # Restrict TensorFlow to only use the first GPU
                tf.config.experimental.set_visible_devices(gpus[gpuID], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[gpuID], True)
            except RuntimeError as e:
                # Visible devices must be set at program startup
                print(e)

    # Build network:
    def load_model(self, model_name, path_weights):
        self.path_weights = path_weights
        self.check_attributes()
        self.net = model_loader.load_model(self.P, self.Ncl, model_name, path_weights)        

    def check_attributes(self):
        self.is_positive_int(self.Ncl, 'Ncl')
        self.is_h5_path(self.path_weights, 'path_weights')
        self.is_multiple_4_int(self.P, 'patch_size')

    def launch(self, dataArray):
        """This function enables to segment a tomogram. As tomograms are too large to be processed in one take, the
        tomogram is decomposed in smaller overlapping 3D patches.

        Args:
            dataArray (3D numpy array): the volume to be segmented
            weights_path (str): path to the .h5 file containing the network weights obtained by the training procedure

        Returns:
            numpy array: contains predicted score maps. Array with index order [class,z,y,x]
        """

        if self.net is None:
            self.load_model(self.dim_in, self.Ncl, 'unet', None)

        self.check_arguments(dataArray, self.P)

        dataArray = (dataArray[:] - np.mean(dataArray[:])) / np.std(dataArray[:])  # normalize
        dataArray = np.pad(dataArray, self.pcrop, mode='constant', constant_values=0)  # zeropad
        dim = dataArray.shape

        l = np.int_(self.P / 2)
        lcrop = np.int_(l - self.pcrop)
        step = np.int_(2 * l + 1 - self.poverlap)

        # Get patch centers:
        pcenterX = list(range(l, dim[0] - l, step))  # list() necessary for py3
        pcenterY = list(range(l, dim[1] - l, step))
        pcenterZ = list(range(l, dim[2] - l, step))

        # If there are still few pixels at the end:
        if pcenterX[-1] < dim[0] - l:
            pcenterX = pcenterX + [dim[0] - l, ]
        if pcenterY[-1] < dim[1] - l:
            pcenterY = pcenterY + [dim[1] - l, ]
        if pcenterZ[-1] < dim[2] - l:
            pcenterZ = pcenterZ + [dim[2] - l, ]

        Npatch = len(pcenterX) * len(pcenterY) * len(pcenterZ)
        self.display('Data array is divided in ' + str(Npatch) + ' patches ...')

        # ---------------------------------------------------------------
        # Process data in patches:
        start = time.time()

        predArray = np.zeros(dim + (self.Ncl,), dtype=np.float16)
        normArray = np.zeros(dim, dtype=np.int8)
        patchCount = 1
        for x in pcenterX:
            for y in pcenterY:
                for z in pcenterZ:
                    self.display('Segmenting patch ' + str(patchCount) + ' / ' + str(Npatch) + ' ...')
                    patch = dataArray[x - l:x + l, y - l:y + l, z - l:z + l]
                    patch = np.reshape(patch, (1, self.P, self.P, self.P, 1))  # reshape for keras [batch,x,y,z,channel]
                    pred = self.net.predict(patch, batch_size=1)

                    predArray[x - lcrop:x + lcrop, y - lcrop:y + lcrop, z - lcrop:z + lcrop, :] = predArray[
                                                                                                  x - lcrop:x + lcrop,
                                                                                                  y - lcrop:y + lcrop,
                                                                                                  z - lcrop:z + lcrop,
                                                                                                  :] + np.float16(pred[0,
                                                                                                       l - lcrop:l + lcrop,
                                                                                                       l - lcrop:l + lcrop,
                                                                                                       l - lcrop:l + lcrop,
                                                                                                       :])
                    normArray[x - lcrop:x + lcrop, y - lcrop:y + lcrop, z - lcrop:z + lcrop] = normArray[
                                                                                               x - lcrop:x + lcrop,
                                                                                               y - lcrop:y + lcrop,
                                                                                               z - lcrop:z + lcrop] + np.ones(
                        (self.P - 2 * self.pcrop, self.P - 2 * self.pcrop, self.P - 2 * self.pcrop), dtype=np.int8)

                    patchCount += 1

        # Normalize overlaping regions:
        for C in range(0, self.Ncl):
            predArray[:, :, :, C] = predArray[:, :, :, C] / normArray

        end = time.time()
        self.display("Model took %0.2f seconds to predict" % (end - start))

        predArray = predArray[self.pcrop:-self.pcrop, self.pcrop:-self.pcrop, self.pcrop:-self.pcrop, :]  # unpad
        return predArray  # predArray is the array containing the scoremaps


    # Similar to function 'segment', only here the tomogram is not decomposed in smaller patches, but processed in one take. However, the tomogram array should be cubic, and the cube length should be a multiple of 4. This function has been developped for tests on synthetic data. I recommend using 'segment' rather than 'segment_single_block'.
    # INPUTS:
    #   dataArray: the volume to be segmented (3D numpy array)
    #   weights_path: path to the .h5 file containing the network weights obtained by the training procedure (string)
    # OUTPUT:
    #   predArray: a numpy array containing the predicted score maps.
    def launch_single_block(self, dataArray):
        self.check_arguments(dataArray, self.P)

        dataArray = (dataArray[:] - np.mean(dataArray[:])) / np.std(dataArray[:])  # normalize
        dataArray = np.pad(dataArray, self.pcrop, mode='constant')  # zeropad
        dim = dataArray.shape
        dataArray = np.reshape(dataArray, (1, dim[0], dim[1], dim[2], 1))  # reshape for keras [batch,x,y,z,channel]

        pred = self.net.predict(dataArray, batch_size=1)
        predArray = pred[0, :, :, :, :]
        #predArray = predArray[self.pcrop:-self.pcrop, self.pcrop:-self.pcrop, self.pcrop:-self.pcrop, :]  # unpad

        return predArray

    def check_arguments(self, dataArray, patch_size):
        self.is_3D_nparray(dataArray, 'tomogram')
        self.check_array_minsize([dataArray, patch_size], ['tomogram', 'patch'])

if __name__ == "__main__":
    cli()
