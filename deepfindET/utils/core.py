# This script is adapted from a public GitHub repository.
# Original source: https://github.com/deep-finder/cryoet-deepfinder/tree/master
# Author: Inria,  Emmanuel Moebel, Charles Kervrann
# License: GPL v3.0

from deepfindET.utils import copick_tools as copicktools
from deepfindET.utils import common as cm
import copick, h5py, os, sys
from itertools import chain
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("agg")  # necessary else: AttributeError: 'NoneType' object has no attribute 'is_interactive'

class DeepFindET:
    def __init__(self):
        self.obs_list = [observer_print]

    # Useful for sending prints to GUI
    def set_observer(self, obs):
        self.obs_list.append(obs)

    # "Master print" calls all observers for prints
    def display(self, message):
        for obs in self.obs_list:
            obs.display(message)

    # For checking inputs:
    def is_3D_nparray(self, v, varname):
        if type(v) != np.ndarray:
            self.display(
                'DeepFindET message: variable "' + varname + '" is ' + str(type(v)) + ". Expected is numpy array.",
            )
            sys.exit()
        if len(v.shape) != 3:
            self.display(
                'DeepFindET message: variable "'
                + varname
                + '" is a '
                + str(len(v.shape))
                + "D array. Expected is a 3D array.",
            )
            sys.exit()

    def is_int(self, v, varname):
        if type(v) != int and type(v) != np.int8 and type(v) != np.int16:
            self.display('DeepFindET message: variable "' + varname + '" is ' + str(type(v)) + ". Expected is int.")
            sys.exit()

    def is_positive_int(self, v, varname):
        self.is_int(v, varname)
        if v <= 0:
            self.display('DeepFindET message: variable "' + varname + '" is <=0. Expected is >0.')
            sys.exit()

    def is_multiple_4_int(self, v, varname):
        self.is_int(v, varname)
        if v % 4 != 0:
            self.display('DeepFindET message: variable "' + varname + '" should be a multiple of 4.')
            sys.exit()

    def is_str(self, v, varname):
        if type(v) != str:
            self.display('DeepFindET message: variable "' + varname + '" is ' + str(type(v)) + ". Expected is str.")
            sys.exit()

    def is_h5_path(self, v, varname):
        self.is_str(v, varname)
        s = os.path.splitext(v)
        if s[1] != ".h5":
            self.display('DeepFindET message: "' + str(varname) + '" points to ' + s[1] + ", expected is .h5")
            sys.exit()

    def is_list(self, v, varname):
        if type(v) != list:
            self.display('DeepFindET message: variable "' + varname + '" is ' + str(type(v)) + ". Expected is list.")
            sys.exit()

    def check_array_minsize(self, v, varname):
        lmin = v[1]  # is expected to be int (e.g. patch length)
        if v[0].shape[0] < lmin and v[0].shape[1] < lmin and v[0].shape[2] < lmin:
            self.display(
                'DeepFindET message: the array "'
                + varname[0]
                + '" has shape '
                + str(v[0].shape)
                + '. Needs to be larger than array "'
                + varname[1]
                + '", which has shape ('
                + str(v[1])
                + ","
                + str(v[1])
                + ","
                + str(v[1])
                + ").",
            )
            sys.exit()

# Following observer classes are needed to send prints to GUI:
class observer_print:
    def display(message):
        print(message)

# Retrieves variable name as a str:
# Found here: https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string
def retrieve_var_name(x, Vars=vars()):
    for k in Vars:
        if type(x) == type(Vars[k]) and x is Vars[k]:
            return k
    return None

# This functions loads the training set at specified paths.
# INPUTS:
#   path_data  : list of strings '/path/to/tomogram.ext'
#   path_target: list of strings '/path/to/target.ext'
#                The idx of above lists correspond to each other so that (path_data[idx], path_target[idx]) corresponds
#                to a (tomog, target) pair
#   dset_name  : can be usefull if files are stored as .h5
# OUTPUTS:
#   data_list  : list of 3D numpy arrays (tomograms)
#   target_list: list of 3D numpy arrays (annotated tomograms)
#                In the same way as for the inputs, (data_list[idx],target_list[idx]) corresponds to a (tomo,target) pair
def load_copick_datasets(copickPath, train_instance, tomoIDs=None):
    data_list = {}
    target_list = {}

    copickRoot = copick.from_file(copickPath)
    if tomoIDs is None:
        tomoIDs = [run.name for run in copickRoot.runs]

    print(f"Loading Targets and Tomograms for the Following Runs: {list(tomoIDs)}")
    for idx in tqdm(range(len(tomoIDs))):
        target_list[tomoIDs[idx]] = copicktools.get_copick_segmentation(
            copickRoot.get_run(tomoIDs[idx]),
            train_instance.labelName,
            train_instance.labelUserID,
            train_instance.sessionID,
        )[:]
        data_list[tomoIDs[idx]] = copicktools.read_copick_tomogram_group(
            copickRoot,
            train_instance.voxelSize,
            train_instance.tomoAlg,
            tomoIDs[idx],
        )[0][:]

        if data_list[tomoIDs[idx]].shape != target_list[tomoIDs[idx]].shape:
            print(f"DeepFinder Message: tomogram and target for run {tomoIDs[idx]} are not of same size!")
            sys.exit()

    return data_list, target_list


# This function applies bootstrap (i.e. re-sampling) in case of unbalanced classes.
# Given an objlist containing objects from various classes, this function outputs an equal amount of objects for each
# class, each objects being uniformely sampled inside its class set.
# INPUTS:
#   objlist: list of dictionaries
#   Nbs    : number of objects to sample from each class
# OUTPUT:
#   bs_idx : list of indexes corresponding to the bootstraped objects
def get_bootstrap_idx(objlist, Nbs):
    # Get a list containing the object class labels (from objlist):
    Nobj = len(objlist)
    label_list = []
    for idx in range(0, Nobj):
        label_list.append(objlist[idx]["label"])

    lblTAB = np.unique(label_list)  # vector containing unique class labels

    # Bootstrap data so that we have equal frequencies (1/Nbs) for all classes:
    # ->from label_list, sample Nbs objects from each class
    bs_idx = []
    for l in lblTAB:
        bs_idx.append(np.random.choice(np.array(np.nonzero(np.array(label_list) == l))[0], Nbs))

    bs_idx = np.concatenate(bs_idx)
    return bs_idx


def query_available_picks(copickRoot, tomoIDs=None, targets=None):
    # Load TomoIDs - Default is Read All TomoIDs from Path
    if tomoIDs is None:
        tomoIDs = [run.name for run in copickRoot.runs]

    labelList = []
    tomoIDList = []
    pickIndList = []
    proteinIndList = []
    proteinCoordsList = []

    for tomoInd in range(len(tomoIDs)):
        copickRun = copickRoot.get_run(tomoIDs[tomoInd])

        if targets is None:
            query = copickRun.picks
        else:
            query = []
            for target_name in targets:
                query += copickRun.get_picks(
                    object_name=target_name,
                    user_id=targets[target_name]["user_id"],
                    session_id=targets[target_name]["session_id"],
                )

        for proteinInd in range(len(query)):
            picks = query[proteinInd]

            nPicks = len(picks.points)
            tomoIDList.append([tomoIDs[tomoInd]] * nPicks)
            pickIndList.append(list(range(nPicks)))
            proteinIndList.append([proteinInd] * nPicks)

            proteinCoordsList.append(picks.points)
            labelList.append([copicktools.get_pickable_object_label(copickRoot, picks.pickable_object_name)] * nPicks)

    labelList = np.array(list(chain.from_iterable(labelList)))
    tomoIDList = np.array(list(chain.from_iterable(tomoIDList)))
    pickIndList = np.array(list(chain.from_iterable(pickIndList)))
    proteinIndList = np.array(list(chain.from_iterable(proteinIndList)))
    proteinCoordsList = np.array(list(chain.from_iterable(proteinCoordsList)))

    return {
        "labelList": labelList,
        "tomoIDlist": tomoIDList,
        "pickIndList": pickIndList,
        "proteinIndList": proteinIndList,
        "proteinCoordsList": proteinCoordsList,
    }


def get_copick_boostrap_idx(organizedPicksDict, Nbs):
    # Bootstrap data so that we have equal frequencies (1/Nbs) for all classes:
    # ->from label_list, sample Nbs objects from each class
    bs_idx = []
    tomoID_idx = []
    pick_idx = []
    protein_idx = []
    protein_picks = []
    lblTAB = np.unique(organizedPicksDict["labelList"])  # vector containing unique class labels
    for l in lblTAB:
        bsIndex = np.random.choice(np.array(np.nonzero(organizedPicksDict["labelList"] == l))[0], Nbs)

        bs_idx.append(bsIndex)
        pick_idx.append(organizedPicksDict["pickIndList"][bsIndex])
        tomoID_idx.append(organizedPicksDict["tomoIDlist"][bsIndex])
        protein_idx.append(organizedPicksDict["proteinIndList"][bsIndex])
        protein_picks.append(organizedPicksDict["proteinCoordsList"][bsIndex])

    return {
        "bs_idx": np.concatenate(bs_idx),
        "tomoID_idx": np.concatenate(tomoID_idx),
        "protein_idx": np.concatenate(protein_idx),
        "pick_idx": np.concatenate(pick_idx),
        "protein_coords": np.concatenate(protein_picks),
    }


# Takes position specified in 'obj', applies random shift to it, and then checks if the patch around this position is
# out of the tomogram boundaries. If so, the position is shifted to that patch is inside the tomo boundaries.
# INPUTS:
#   tomodim: tuple (dimX,dimY,dimZ) containing size of tomogram
#   p_in   : int lenght of patch in voxels
#   obj    : dictionary obtained when calling objlist[idx]
#   Lrnd   : int random shift in voxels applied to position
# OUTPUTS:
#   x,y,z  : int,int,int coordinates for sampling patch safely
def get_copick_patch_position(tomodim, p_in, Lrnd, voxelSize, copicks):
    # sample at coordinates specified in obj=objlist[idx]
    x = int(copicks.location.x / voxelSize)
    y = int(copicks.location.y / voxelSize)
    z = int(copicks.location.z / voxelSize)

    x,y,z = add_random_shift(tomodim,p_in,Lrnd,x,y,z)

    return x, y, z

def add_random_shift(tomodim, p_in, Lrnd, x,y,z):

    # Add random shift to coordinates:
    x = x + np.random.choice(range(-Lrnd,Lrnd+1))
    y = y + np.random.choice(range(-Lrnd,Lrnd+1))
    z = z + np.random.choice(range(-Lrnd,Lrnd+1))
    
    # Shift position if too close to border:
    if (x<p_in) : x = p_in
    if (y<p_in) : y = p_in
    if (z<p_in) : z = p_in
    if (x>tomodim[2]-p_in): x = tomodim[2]-p_in
    if (y>tomodim[1]-p_in): y = tomodim[1]-p_in
    if (z>tomodim[0]-p_in): z = tomodim[0]-p_in

    #else: # sample random position in tomogram
    #    x = np.int32( np.random.choice(range(p_in,tomodim[0]-p_in)) )
    #    y = np.int32( np.random.choice(range(p_in,tomodim[0]-p_in)) )
    #    z = np.int32( np.random.choice(range(p_in,tomodim[0]-p_in)) )
    
    return x,y,z 


# Saves training history as .h5 file.
# INPUTS:
#   history: dictionary object containing lists. These lists contain scores and metrics wrt epochs.
#   filename: string '/path/to/net_train_history.h5'
def save_history(history, filename):
    if os.path.isfile(filename):  # if file exists, delete before writing the updated version
        os.remove(filename)  # quick fix for OSError: Can't write data (no appropriate function for conversion path)

    h5file = h5py.File(filename, "w")

    # train and val loss & accuracy:
    dset = h5file.create_dataset("acc", np.array(history["acc"]).shape, dtype="float16")
    dset[:] = np.array(history["acc"], dtype="float16")
    dset = h5file.create_dataset("loss", np.array(history["loss"]).shape, dtype="float16")
    dset[:] = np.array(history["loss"], dtype="float16")
    dset = h5file.create_dataset("val_acc", np.array(history["val_acc"]).shape, dtype="float16")
    dset[:] = np.array(history["val_acc"], dtype="float16")
    dset = h5file.create_dataset("val_loss", np.array(history["val_loss"]).shape, dtype="float16")
    dset[:] = np.array(history["val_loss"], dtype="float16")

    # val precision, recall, F1:
    dset = h5file.create_dataset("val_f1", np.array(history["val_f1"]).shape, dtype="float16")
    dset[:] = np.array(history["val_f1"], dtype="float16")
    dset = h5file.create_dataset("val_precision", np.array(history["val_precision"]).shape, dtype="float16")
    dset[:] = np.array(history["val_precision"], dtype="float16")
    dset = h5file.create_dataset("val_recall", np.array(history["val_recall"]).shape, dtype="float16")
    dset[:] = np.array(history["val_recall"], dtype="float16")

    h5file.close()
    return


def read_history(filename, save_fig=True):
    history = {
        "acc": None,
        "loss": None,
        "val_acc": None,
        "val_loss": None,
        "val_f1": None,
        "val_precision": None,
        "val_recall": None,
    }

    h5file = h5py.File(filename, "r")
    # train and val loss & accuracy:
    history["acc"] = h5file["acc"][:]
    history["loss"] = h5file["loss"][:]
    history["val_acc"] = h5file["val_acc"][:]
    history["val_loss"] = h5file["val_loss"][:]
    # val precision, recall, F1:
    history["val_f1"] = h5file["val_f1"][:]
    history["val_precision"] = h5file["val_precision"][:]
    history["val_recall"] = h5file["val_recall"][:]

    h5file.close()
    return history


# Plots the training history as several graphs and saves them in an image file.
# Validation score is averaged over all batches tested in validation step (steps_per_valid)
# Training score is averaged over last N=steps_per_valid batches of each epoch.
#   -> This is to have similar curve smoothness to validation.
# INPUTS:
#   history: dictionary object containing lists. These lists contain scores and metrics wrt epochs.
#   filename: string '/path/to/net_train_history_plot.png'
def plot_history(history, 
                filename:str = 'net_train_history.png', 
                save_figure: bool = True):

    Ncl = len(history["val_f1"][0])
    legend_names = []
    for lbl in range(0, Ncl):
        legend_names.append("class " + str(lbl))

    len(history["val_loss"])

    hist_loss_train = history["loss"]
    hist_acc_train = history["acc"]
    hist_loss_valid = history["val_loss"]
    hist_acc_valid = history["val_acc"]
    hist_f1 = history["val_f1"]
    hist_recall = history["val_recall"]
    hist_precision = history["val_precision"]

    fig = plt.figure(figsize=(15, 12))
    plt.subplot(321)
    plt.plot(hist_loss_train, label="train")
    plt.plot(hist_loss_valid, label="valid")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.legend()
    plt.grid()

    plt.subplot(323)
    plt.plot(hist_acc_train, label="train")
    plt.plot(hist_acc_valid, label="valid")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend()
    plt.grid()

    plt.subplot(322)
    plt.plot(hist_f1)
    plt.ylabel("F1-score")
    plt.xlabel("epochs")
    plt.legend(legend_names)
    plt.grid()

    plt.subplot(324)
    plt.plot(hist_precision)
    plt.ylabel("precision")
    plt.xlabel("epochs")
    plt.grid()

    plt.subplot(326)
    plt.plot(hist_recall)
    plt.ylabel("recall")
    plt.xlabel("epochs")
    plt.grid()

    if save_figure:
        fig.savefig(filename)

def convert_hdf5_to_dictionary(filename: str):
    """
    Converts an HDF5 file into a nested dictionary. Each group in the HDF5 file becomes a 
    nested dictionary, and datasets are converted to NumPy arrays.

    Parameters:
    filename (str): Path to the HDF5 file to be converted.

    Returns:
    dict: A dictionary representation of the HDF5 file contents.
    """

    with h5py.File(filename, 'r') as hdf:
        def recursively_load_dict_contents(hdf_group):
            """Recursively loads the HDF5 group into a nested dictionary."""
            ans = {}
            for key, item in hdf_group.items():
                # Check if the item is a dataset and convert it to a NumPy array
                if isinstance(item, h5py.Dataset):
                    ans[key] = item[()]  # Get the dataset as a NumPy array
                # If the item is another group, recurse into it and convert it to a dictionary
                elif isinstance(item, h5py.Group):
                    ans[key] = recursively_load_dict_contents(item)
            return ans
        
        # Convert the HDF5 file contents into a dictionary
        history_dict = recursively_load_dict_contents(hdf)

    # Return the fully constructed dictionary
    return history_dict
