from scipy.ndimage import rotate, zoom, map_coordinates, gaussian_filter
import os, warnings, random, time
import numpy as np

class DataAugmentation:
    """
    Class for applying various data augmentation techniques to 3D tomograms and their corresponding targets.
    """

    def __init__(self, seed=None):
        """
        Initialize the DataAugmentation class with a list of augmentation functions.
        """
        
        # TODO: Add Random Noise 
        self.augmentations = [
            self.brightness,
            self.gaussian_blur,
            self.intensity_scaling,
            self.contrast_adjustment, 
            self.rotation_180_degrees,
            # self.angle_rotation,            
        ]

        self.axes = [(0, 1), (0, 2), (1, 2)]

        # Set the seed for the random number generators. 
        # If no seed is provided, use the current time.
        if seed is None:
            seed = int(time.time())
        np.random.seed(seed)
        random.seed(seed)            


    def apply_augmentations(self, volume, target):
        """
        Apply a random sequence of augmentations to the given volume and target.

        Parameters:
        volume (numpy.ndarray): The input 3D volume.
        target (numpy.ndarray): The corresponding target mask.

        Returns:
        tuple: The augmented volume and target.
        """
        random.shuffle(self.augmentations)
        for augmentation in self.augmentations:
            if augmentation == self.angle_rotation or augmentation == self.rotation_180_degrees:
                volume, target = augmentation(volume, target)
            else:
                volume = augmentation(volume)

        return volume, target

    ####################### Intensity Transformations  #######################

    def brightness(self, volume, max_delta=0.2):
        """
        Parameters:
        volume (numpy.ndarray): The input 3D volume.
        max_delta (float): The maximum change in brightness.

        Returns:
        numpy.ndarray: The brightness adjusted volume.
        """        
        delta = np.random.uniform(-max_delta, max_delta)
        volume += delta
        return volume

    def gaussian_blur(self, volume, sigma_range=(0.5, 1.5)):
        """
        Parameters:
        volume (numpy.ndarray): The input 3D volume.
        sigma_range (tuple): The range of sigma values for the Gaussian kernel.

        Returns:
        numpy.ndarray: The blurred volume.
        """
        sigma = random.uniform(*sigma_range)
        blurred_volume = gaussian_filter(volume, sigma=sigma)
        return blurred_volume    

    def intensity_scaling(self, volume, intensity_range=(0.8, 1.2)):
        """
        Parameters:
        volume (numpy.ndarray): The input 3D volume.
        intensity_range (tuple): The range of intensity scaling factors.

        Returns:
        numpy.ndarray: The intensity scaled volume.
        """        
        intensity_factor = np.random.uniform(*intensity_range)
        scaled_volume = volume * intensity_factor
        return scaled_volume

    def contrast_adjustment(self, volume, contrast_range=(0.8, 1.2)):
        """
        Parameters:
        volume (numpy.ndarray): The input 3D volume.
        contrast_range (tuple): The range of contrast adjustment factors.

        Returns:
        numpy.ndarray: The contrast adjusted volume.
        """        
        contrast_factor = np.random.uniform(*contrast_range)
        mean = np.mean(volume)
        adjusted_volume = mean + contrast_factor * (volume - mean)
        return adjusted_volume    

    def scaling(self, volume, target, scale_range=(0.95, 1.05)):
        """
        Parameters:
        volume (numpy.ndarray): The input 3D volume.
        target (numpy.ndarray): The corresponding target mask.
        scale_range (tuple): The range of scaling factors.

        Returns:
        tuple: The scaled volume and target.
        """        
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        volume = zoom(volume, scale_factor)
        target = zoom(target, scale_factor, order=0)
        return volume, target

    # Additive Noise

    ####################### Geometric Transformations  #######################

    def rotation_180_degrees(self, volume, target, augment_probability=0.5):

        # Apply fixed 180-degree rotation with some probability
        if np.random.rand() < augment_probability:

            # chosen_axis = self.axes[np.random.randint(0, len(self.axes))]
            chosen_axis = (0,2)

            # Rotate by 180 degrees around the x-z plane
            volume = np.rot90(volume, k=2, axes=chosen_axis)
            target = np.rot90(target, k=2, axes=chosen_axis)
        
        return volume, target


    def angle_rotation(self, volume, target, max_angle=15, augment_probability=0.8):
        """
        Parameters:
        volume (numpy.ndarray): The input 3D volume.
        target (numpy.ndarray): The corresponding target mask.
        max_angle (float): The maximum rotation angle in degrees.

        Returns:
        tuple: The rotated volume and target.
        """

        if np.random.rand() < augment_probability:
            angle = np.random.uniform(-max_angle, max_angle)
            chosen_axis = self.axes[np.random.randint(0, len(self.axes))]
            volume = rotate(volume, angle, axes=chosen_axis, reshape=False, mode='reflect')
            target = rotate(target, angle, axes=chosen_axis, reshape=False, mode='reflect')
        return volume, target   

    def elastic_transform(self, volume, target, alpha=15, sigma=3):
        """
        Parameters:
        volume (numpy.ndarray): The input 3D volume.
        target (numpy.ndarray): The corresponding target mask.
        alpha (float): The scaling factor for deformation.
        sigma (float): The standard deviation for Gaussian filter.

        Returns:
        tuple: The elastically transformed volume and target.
        """        
        random_state = np.random.RandomState(None)
        shape = volume.shape

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z + dz, (-1, 1))

        distorted_volume = map_coordinates(volume, indices, order=1, mode='reflect').reshape(shape)
        distorted_target = map_coordinates(target, indices, order=0, mode='reflect').reshape(shape)

        return distorted_volume, distorted_target        
