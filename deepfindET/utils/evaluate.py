from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import pairwise_distances
from deepfindET.utils import objl as ol
from scipy.spatial import distance
from pycm import ConfusionMatrix
import matplotlib.pyplot as plt
import numpy as np
import copy

def remove_repeated_picks_v2(coordinates, distanceThreshold, pixelSize = 1):

    # Calculate the distance matrix for the 3D coordinates
    dist_matrix = distance.cdist(coordinates[:, :3]/pixelSize, coordinates[:, :3]/pixelSize)

    # Create a linkage matrix using single linkage method
    Z = linkage(dist_matrix, method='complete')

    # Form flat clusters with a distance threshold to determine groups
    clusters = fcluster(Z, t=distanceThreshold, criterion='distance')

    # Initialize an array to store the average of each group
    unique_coordinates = np.zeros((max(clusters), coordinates.shape[1]))

    # Calculate the mean for each cluster
    for i in range(1, max(clusters) + 1):
        unique_coordinates[i-1] = np.mean(coordinates[clusters == i], axis=0)

    return unique_coordinates    

def remove_repeated_picks(coordinates, distanceThreshold, pixelSize = 1):

    # Compute pairwise distances
    dist_matrix = pairwise_distances(coordinates)
    
    # Use hierarchical clustering to group close points
    distanceThreshold = distanceThreshold**2
    linkage_matrix = linkage(dist_matrix, method='single', metric='euclidean')
    labels = fcluster(linkage_matrix, t=distanceThreshold, criterion='distance')
    
    # Calculate the average coordinates for each group
    unique_labels = np.unique(labels)
    filteredCoordinates = np.array([coordinates[labels == label].mean(axis=0) for label in unique_labels])

    return filteredCoordinates 

def compute_metrics(gt_points, pred_points, threshold):
    gt_points = np.array(gt_points)
    pred_points = np.array(pred_points)
    
    # Calculate distances
    dist_matrix = distance.cdist(pred_points, gt_points, 'euclidean')

    # Determine matches within the threshold
    tp = np.sum(np.min(dist_matrix, axis=1) < threshold)
    fp = np.sum(np.min(dist_matrix, axis=1) >= threshold)
    fn = np.sum(np.min(dist_matrix, axis=0) >= threshold)
    
    # Precision, Recall, F1 Score
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    accuracy = tp / (tp + fp + fn)  # Note: TN not considered here
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy, 
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }

class Evaluator:

    def __init__(self, dset_true, dset_pred, dist_thr):
        if dset_true.keys() != dset_pred.keys():
            raise Exception('Data sets do not have the same keys')
        self.dset_true = dset_true
        self.dset_pred = dset_pred
        self.dist_thr = dist_thr

        self.tomoid_keys = dset_true.keys()

        # detect_eval is a dict who stores, per tomo, all info related to the evaluation of the detections
        self.detect_eval = {}

        # Initialize detect_eval:
        for tomoid in dset_true.keys():
            self.detect_eval[tomoid] = {
                'objl_true': None,
                'objl_pred': None,
                'dmat': None,
                'cmat': None,
                'objl_tp': None,
                'objl_fp': None,
                'objl_fn': None,
                'pre': None,
                'rec': None,
                'f1s': None,
            }
        self.detect_eval['global'] = {
            'cmat': None,
            'pre': None,
            'rec': None,
            'f1s': None,
            'n_tp': None,
            'n_fp': None,
            'n_fn': None,
        }

        # detect_eval_list stores all the detect_eval w.r.t score_thr_list
        self.detect_eval_list = []

    def get_evaluation_wrt_detection_score(self, score_thr_list):
        for score_thr in score_thr_list:
            self.get_evaluation(score_thr)
            self.detect_eval_list.append(copy.deepcopy(self.detect_eval))

        return self.detect_eval_list

    def get_evaluation(self, score_thr=None):
        for tomoid in self.tomoid_keys:
            objl_true = self.dset_true[tomoid]['object_list']
            objl_pred = self.dset_pred[tomoid]['object_list']
            if score_thr is not None:
                objl_pred = ol.above_thr(objl_pred, score_thr)
            self.detect_eval[tomoid]['objl_true'] = objl_true
            self.detect_eval[tomoid]['objl_pred'] = objl_pred

            self.get_distance_matrix(tomoid)

        self.get_confusion_matrix()
        self.get_metrics()

        return self.detect_eval

    def get_distance_matrix(self, tomoid):
        # Prepare data points for pairwise_distances:
        objl_true = self.detect_eval[tomoid]['objl_true']
        objl_pred = self.detect_eval[tomoid]['objl_pred']

        coords_true = np.zeros((len(objl_true), 3))
        coords_pred = np.zeros((len(objl_pred), 3))
        for idx, obj in enumerate(objl_true):
            coords_true[idx, 0] = obj['y']
            coords_true[idx, 1] = obj['x']
            coords_true[idx, 2] = obj['z']
        for idx, obj in enumerate(objl_pred):
            coords_pred[idx, 0] = obj['y']
            coords_pred[idx, 1] = obj['x']
            coords_pred[idx, 2] = obj['z']

        # Compute pairwise distances:
        if len(objl_pred) is not 0:  # only compute dmat if something has been detected
            dmat = pairwise_distances(coords_true, coords_pred, metric='euclidean')
        else:  # if no detections then dmat is not defined (and later on all scores are 0)
            dmat = None

        self.detect_eval[tomoid]['dmat'] = dmat

    def get_confusion_matrix(self):
        y_true_global = []
        y_pred_global = []
        for tomoid in self.tomoid_keys:
            dmat = self.detect_eval[tomoid]['dmat']
            objl_true = self.detect_eval[tomoid]['objl_true'].copy()
            objl_pred = self.detect_eval[tomoid]['objl_pred'].copy()
            if dmat is not None:  # i.e., if something has been detected
                # The aim of this part is to construct objl_pred_corresp and objl_true_corresp
                # from objl_pred and objl_true, such that objl_pred_corresp[i] matches objl_true_corresp[i]
                # (matches = same position with a tolerated position error of self.dist_thr)

                # Initialize objl_pred_corresp
                objl_pred_corresp = [None for _ in range(len(objl_true))]
                objl_true_corresp = objl_true.copy()

                # Correspondence matrix where '1' means that an entry from pred is situated at a distance <=dist_thr
                # to an entry of true:
                corresp_mat = dmat <= self.dist_thr

                pred_match_idx_list = []  # this list stores all the entries in objl_pred that have a match in objl_true
                for idx, obj in enumerate(objl_true_corresp):
                    indices = np.nonzero(corresp_mat[idx, :])[0]
                    if len(indices) == 1:  # tp only if 1 corresponence (necessary in case 1 true matches several pred)
                        objl_pred_corresp[idx] = objl_pred[indices[0]]
                        pred_match_idx_list.append(indices[0])
                    elif len(indices) < 1 or len(indices) > 1:  # if no correspondence or multiple correspondence, then fp
                        obj_fp = copy.deepcopy(obj)  # necessary else the labels in objl_true get modified
                        obj_fp['label'] = 0  # fp, therefore label is 0
                        objl_pred_corresp[idx] = obj_fp
                    else:
                        print('Exception!! This case should never happen.')

                # First, get objl_tp:
                pred_match_idx_list = np.unique(pred_match_idx_list)  # necessary in case 1 pred matches several true. Therefore 1 pred is counted only once as tp
                pred_match_idx_list = np.flip(np.sort(pred_match_idx_list))  # sort idx in descending order, else it is a mess when deleting elements

                # Get all predictions that have a match in objl_true (i.e., objl_tp):
                objl_tp = []
                for idx in pred_match_idx_list:
                    objl_tp.append(objl_pred[idx])

                # Get all predictions that do not have a match in objl_true:
                objl_pred_no_corresp = copy.deepcopy(objl_pred) # initialize
                for idx in pred_match_idx_list:
                    del objl_pred_no_corresp[idx]

                # All obj in objl_pred_no_corresp are false positives. So we add them in objl_true_no_corresp with label 0
                objl_true_no_corresp = copy.deepcopy(objl_pred_no_corresp)
                for idx, obj in enumerate(objl_true_no_corresp):
                    obj['label'] = 0

                # Finally:
                objl_true_corresp = objl_true_corresp + objl_true_no_corresp
                objl_pred_corresp = objl_pred_corresp + objl_pred_no_corresp

                # Now it is easy to get objl_fp and objl_fn:
                objl_fp = ol.get_class(objl_true_corresp, label=0)
                objl_fn = ol.get_class(objl_pred_corresp, label=0)

                # Prepare the inputs of ConfusionMatrix:
                y_true = [obj['label'] for obj in objl_true_corresp]
                y_pred = [obj['label'] for obj in objl_pred_corresp]

            else:  # if nothing has been detected
                objl_true_corresp = objl_true.copy()
                y_true = [obj['label'] for obj in objl_true_corresp]
                y_pred = [0 for _ in range(len(y_true))]

                objl_tp = []
                objl_fp = []
                objl_fn = objl_true.copy()

            # Get ConfusionMatrix:
            cmat = ConfusionMatrix(actual_vector=y_true, predict_vector=y_pred)

            # Store:
            self.detect_eval[tomoid]['cmat'] = cmat
            self.detect_eval[tomoid]['objl_tp'] = objl_tp
            self.detect_eval[tomoid]['objl_fp'] = objl_fp
            self.detect_eval[tomoid]['objl_fn'] = objl_fn

            y_true_global += y_true
            y_pred_global += y_pred

        cmat_global = ConfusionMatrix(actual_vector=y_true_global, predict_vector=y_pred_global)
        self.detect_eval['global']['cmat'] = cmat_global

    def get_metrics(self):
        n_true_global = 0
        n_pred_global = 0
        for tomoid in self.tomoid_keys:
            self.detect_eval[tomoid]['pre'] = self.detect_eval[tomoid]['cmat'].PPV
            self.detect_eval[tomoid]['rec'] = self.detect_eval[tomoid]['cmat'].TPR
            self.detect_eval[tomoid]['f1s'] = self.detect_eval[tomoid]['cmat'].F1
            self.detect_eval[tomoid]['n_tp'] = self.detect_eval[tomoid]['cmat'].TP
            self.detect_eval[tomoid]['n_fp'] = self.detect_eval[tomoid]['cmat'].FP
            self.detect_eval[tomoid]['n_fn'] = self.detect_eval[tomoid]['cmat'].FN

            n_true = len(self.detect_eval[tomoid]['objl_true'])
            n_pred = len(self.detect_eval[tomoid]['objl_pred'])
            self.detect_eval[tomoid]['n_true'] = n_true
            self.detect_eval[tomoid]['n_pred'] = n_pred

            n_true_global += n_true
            n_pred_global += n_pred


        self.detect_eval['global']['pre'] = self.detect_eval['global']['cmat'].PPV
        self.detect_eval['global']['rec'] = self.detect_eval['global']['cmat'].TPR
        self.detect_eval['global']['f1s'] = self.detect_eval['global']['cmat'].F1
        self.detect_eval['global']['n_tp'] = self.detect_eval['global']['cmat'].TP
        self.detect_eval['global']['n_fp'] = self.detect_eval['global']['cmat'].FP
        self.detect_eval['global']['n_fn'] = self.detect_eval['global']['cmat'].FN

        self.detect_eval['global']['n_true'] = n_true_global
        self.detect_eval['global']['n_pred'] = n_pred_global

    def to_labelmap(self, scoremaps):
        """Converts scoremaps into a labelmap.

        Args:
            scoremaps (4D numpy array): array with index order [class,z,y,x]

        Returns:
            3D numpy array: array with index order [z,y,x]
        """
        labelmap = np.int8( np.argmax(scoremaps,3) )
        return labelmap


def plot_eval(detect_eval, class_label, score_thr_list):
    pre_list = []
    rec_list = []
    f1s_list = []
    for deval in detect_eval:
        pre = deval['global']['pre'][class_label]
        rec = deval['global']['rec'][class_label]
        f1s = deval['global']['f1s'][class_label]

        pre_list.append(pre)
        rec_list.append(rec)
        f1s_list.append(f1s)

    # Get max(F1-score) with corresponding precision and recall:
    f1s_max = np.max(f1s_list)
    idx_max = np.argmax(f1s_list)
    pre_best = pre_list[idx_max]
    rec_best = rec_list[idx_max]

    f1s_max = np.round(f1s_max, 2)
    pre_best = np.round(pre_best, 2)
    rec_best = np.round(rec_best, 2)

    # Drop the plot:
    fontsize = 10
    fig, ax = plt.subplots(1, 1)
    ax.plot(score_thr_list, pre_list)
    ax.plot(score_thr_list, rec_list)
    ax.plot(score_thr_list, f1s_list)
    ax.set_title(f'max(f1-score)={f1s_max}, [pre,rec]=[{pre_best},{rec_best}]', fontsize=fontsize)
    ax.set_xlim([np.min(score_thr_list), np.max(score_thr_list)])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Detection score threshold')
    ax.set_ylabel('Metrics')
    ax.grid(True)
    ax.legend(['Precision', 'Recall', 'F1-score'])

    return fig
