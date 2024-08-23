from . import common as cm
from . import core
import numpy as np

class TargetBuilder(core.DeepFinder):
    def __init__(self):
        core.DeepFinder.__init__(self)

        self.remove_flag = False # if true, places '0' at object voxels, instead of 'lbl'.
                                 # Usefull in annotation tool, for removing objects from target

    # Generates segmentation targets from object list. Here macromolecules are annotated with their shape.
    # INPUTS
    #   objl: list of dictionaries. Needs to contain [phi,psi,the] Euler angles for orienting the shapes.
    #   target_array: 3D numpy array that initializes the training target. Allows to pass an array already containing annotated structures like membranes.
    #                 index order of array should be [z,y,x]
    #   ref_list: list of binary 3D arrays (expected to be cubic). These reference arrays contain the shape of macromolecules ('1' for 'is object' and '0' for 'is not object')
    #             The references order in list should correspond to the class label
    #             For ex: 1st element of list -> reference of class 1
    #                     2nd element of list -> reference of class 2 etc.
    # OUTPUT
    #   target_array: 3D numpy array. '0' for background class, {'1','2',...} for object classes.
    def generate_with_shapes(self, objl, target_array, ref_list):
        """Generates segmentation targets from object list. Here macromolecules are annotated with their shape.

        Args:
            objl (list of dictionaries): Needs to contain [phi,psi,the] Euler angles for orienting the shapes.
            target_array (3D numpy array): array that initializes the training target. Allows to pass an array already containing annotated structures like membranes.
                index order of array should be [z,y,x]
            ref_list (list of 3D numpy arrays): These reference arrays are expected to be cubic and to contain the shape of macromolecules ('1' for 'is object' and '0' for 'is not object')
                The references order in list should correspond to the class label.
                For ex: 1st element of list -> reference of class 1; 2nd element of list -> reference of class 2 etc.

        Returns:
            3D numpy array: Target array, where '0' for background class, {'1','2',...} for object classes.
        """
        self.check_arguments(objl, target_array, ref_list)

        N = len(objl)
        dim = target_array.shape
        for p in range(len(objl)):
            self.display('Annotating object ' + str(p + 1) + ' / ' + str(N) + ' ...')
            lbl = int(objl[p]['label'])
            x = int(objl[p]['x'])
            y = int(objl[p]['y'])
            z = int(objl[p]['z'])
            phi = objl[p]['phi']
            psi = objl[p]['psi']
            the = objl[p]['the']

            ref = ref_list[lbl - 1]
            centeroffset = np.int_(np.floor(ref.shape[0] / 2)) # here we expect ref to be cubic

            # Rotate ref:
            if phi!=None and psi!=None and the!=None:
                ref = cm.rotate_array(ref, (phi, psi, the))
                ref = np.int8(np.round(ref))

            # Get the coordinates of object voxels in target_array
            obj_voxels = np.nonzero(ref == 1)
            x_vox = obj_voxels[2] + x - centeroffset #+1
            y_vox = obj_voxels[1] + y - centeroffset #+1
            z_vox = obj_voxels[0] + z - centeroffset #+1

            for idx in range(x_vox.size):
                xx = x_vox[idx]
                yy = y_vox[idx]
                zz = z_vox[idx]
                if xx >= 0 and xx < dim[2] and yy >= 0 and yy < dim[1] and zz >= 0 and zz < dim[0]:  # if in tomo bounds
                    if self.remove_flag:
                        target_array[zz, yy, xx] = 0
                    else:
                        target_array[zz, yy, xx] = lbl

        return np.int8(target_array)

    def check_arguments(self, objl, target_array, ref_list):
        self.is_list(objl, 'objl')
        self.is_3D_nparray(target_array, 'target_array')
        self.is_list(ref_list, 'ref_list')

    # Generates segmentation targets from object list. Here macromolecules are annotated with spheres.
    # This method does not require knowledge of the macromolecule shape nor Euler angles in the objl.
    # On the other hand, it can be that a network trained with 'sphere targets' is less accurate than with 'shape targets'
    # INPUTS
    #   objl: list of dictionaries.
    #   target_array: 3D numpy array that initializes the training target. Allows to pass an array already containing annotated structures like membranes.
    #                 index order of array should be [z,y,x]
    #   radius_list: list of sphere radii (in voxels).
    #             The radii order in list should correspond to the class label
    #             For ex: 1st element of list -> sphere radius for class 1
    #                     2nd element of list -> sphere radius for class 2 etc.
    # OUTPUT
    #   target_array: 3D numpy array. '0' for background class, {'1','2',...} for object classes.
    def generate_with_spheres(self, objl, target_array, radius_list):
        """Generates segmentation targets from object list. Here macromolecules are annotated with spheres.
        This method does not require knowledge of the macromolecule shape nor Euler angles in the objl.
        On the other hand, it can be that a network trained with 'sphere targets' is less accurate than with 'shape targets'.

        Args:
            objl (list of dictionaries)
            target_array (3D numpy array): array that initializes the training target. Allows to pass an array already containing annotated structures like membranes.
                index order of array should be [z,y,x]
            radius_list (list of int): contains sphere radii per class (in voxels).
                The radii order in list should correspond to the class label.
                For ex: 1st element of list -> sphere radius for class 1, 2nd element of list -> sphere radius for class 2 etc.

        Returns:
            3D numpy array: Target array, where '0' for background class, {'1','2',...} for object classes.
        """
        Rmax = max(radius_list)
        dim = [2*Rmax, 2*Rmax, 2*Rmax]
        ref_list = []
        for idx in range(len(radius_list)):
            ref_list.append(cm.create_sphere(dim, radius_list[idx]))
        target_array = self.generate_with_shapes(objl, target_array, ref_list)
        return target_array


