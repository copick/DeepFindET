from collections import defaultdict
import tensorflow as tf
import copick, json
import numpy as np

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import Adam

from deepfindET import callbacks, losses, settings
from deepfindET.utils import core, augmentdata
from deepfindET.models import model_loader

policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)

# TODO: add method for resuming training. It should load existing weights and train_history. So when restarting, the plot curves show prececedent epochs
class Train(core.DeepFindET):
    def __init__(self, Ncl, dim_in):
        core.DeepFindET.__init__(self)
        self.path_out = "./"

        # Network parameters:
        self.Ncl = Ncl  # Number of Classes
        self.dim_in = dim_in  # /!\ has to a multiple of 4 (because of 2 pooling layers), so that dim_in=dim_out
        
        # Initialize Empty network:
        self.net = None

        self.label_list = []
        for l in range(self.Ncl):
            self.label_list.append(l)  # for precision_recall_fscore_support
        # (else bug if not all labels exist in batch)

        # Training parameters:
        self.batch_size = 25
        self.epochs = 100
        self.steps_per_epoch = 100
        self.steps_per_valid = 10  # number of samples for validation

        # Optimization Paramters 
        self.learning_rate = 0.0001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epislon = 1e-8
        self.decay = 0
        self.optimizer = Adam(learning_rate=self.learning_rate, 
                              beta_1=self.beta1, beta_2=self.beta2, 
                              epsilon=self.epislon, decay=self.decay)
        self.loss = losses.tversky_loss

        # random shifts applied when sampling data- and target-patches (in voxels)
        self.Lrnd = 15

        self.class_weights = None
        self.sample_weights = None  # np array same lenght as objl_train
        self.trainTomoIDs = None
        self.validTomoIDs = None
        self.targets = None

        self.flag_batch_bootstrap = 1

        self.gpuID = None

        self.NsubEpoch = 10
        self.sample_size = 15

        self.check_attributes()
        self.data_augmentor = augmentdata.DataAugmentation()

    def check_attributes(self):
        self.is_positive_int(self.Ncl, "Ncl")
        self.is_multiple_4_int(self.dim_in, "dim_in")
        self.is_positive_int(self.batch_size, "batch_size")
        self.is_positive_int(self.epochs, "epochs")
        self.is_positive_int(self.steps_per_epoch, "steps_per_epoch")
        self.is_positive_int(self.steps_per_valid, "steps_per_valid")
        self.is_int(self.Lrnd, "Lrnd")

    def create_class_weights(self, input_class_weights, copick_path):

        copickRoot = copick.from_file(copick_path)

        self.class_weights = defaultdict(lambda: 1, {i: 1 for i in range(self.Ncl)})
        for weights in input_class_weights:
            self.class_weights[copickRoot.get_object(weights[0]).label] = weights[1]        

    def load_model(self, model_name, trained_weights_path = None):
        self.net, self.model_parameters = model_loader.load_model(self.dim_in, self.Ncl, model_name, trained_weights_path)        

    # This function launches the training procedure. For each epoch, an image is plotted, displaying the progression
    # with different metrics: loss, accuracy, f1-score, recall, precision. Every 10 epochs, the current network weights
    # are saved.
    # INPUTS:
    #   path_data     : a list containing the paths to data files (i.e. tomograms)
    #   path_target   : a list containing the paths to target files (i.e. annotated volumes)
    #   objlist_train : list of dictionaries containing information about annotated objects (e.g. class, position)
    #                   In particular, the tomo_idx should correspond to the index of 'path_data' and 'path_target'.
    #                   See utils/objl.py for more info about object lists.
    #                   During training, these coordinates are used for guiding the patch sampling procedure.
    #   objlist_valid : same as 'objlist_train', but objects contained in this list are not used for training,
    #                   but for validation. It allows to monitor the training and check for over/under-fitting. Ideally,
    #                   the validation objects should originate from different tomograms than training objects.
    # The network is trained on small 3D patches (i.e. sub-volumes), sampled from the larger tomograms (due to memory
    # limitation). The patch sampling is not realized randomly, but is guided by the macromolecule coordinates contained
    # in so-called object lists (objlist).
    def launch(self, path_train, path_valid=None):
        """This function launches the training procedure. For each epoch, an image is plotted, displaying the progression
        with different metrics: loss, accuracy, f1-score, recall, precision. Every 10 epochs, the current network weights
        are saved.

        Args:
            path_data (list of string): contains paths to data files (i.e. tomograms)
            path_target (list of string): contains paths to target files (i.e. annotated volumes)
            objlist_train (list of dictionaries): contains information about annotated objects (e.g. class, position)
                In particular, the tomo_idx should correspond to the index of 'path_data' and 'path_target'.
                See utils/objl.py for more info about object lists.
                During training, these coordinates are used for guiding the patch sampling procedure.
            objlist_valid (list of dictionaries): same as 'objlist_train', but objects contained in this list are not
                used for training, but for validation. It allows to monitor the training and check for over/under-fitting.
                Ideally, the validation objects should originate from different tomograms than training objects.

        Note:
            The function saves following files at regular intervals:
                net_weights_epoch*.h5: contains current network weights

                net_train_history.h5: contains arrays with all metrics per training iteration

                net_train_history_plot.png: plotted metric curves

        """
        self.check_attributes()

        if self.net is None:
            self.net, self.model_parameters = model_loader.load_model(self.dim_in, self.Ncl, 'unet', None)          

        # TensorBoard writer
        log_dir = self.path_out + "tensorboard_logs/"
        tf.summary.create_file_writer(log_dir)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=0)

        # gpus = tf.config.list_logical_devices('GPU')
        # strategy = tf.distribute.MirroredStrategy(gpus)

        # Check GPU memory limit
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                # Set the Desired GPU if Multi-Model Training is Specified
                if self.gpuID is not None:
                    tf.config.experimental.set_visible_devices(gpus[self.gpuID], "GPU")
                # Enable memory growth for the GPU
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

            except RuntimeError as e:
                print(e)
        else:
            print("No GPU found.")        

        # Build network (not in constructor, else not possible to init model with weights from previous train round):
        self.net.compile(optimizer=self.optimizer, loss=self.loss, metrics=["accuracy"])

        self.batch_data = np.zeros((self.batch_size, self.dim_in, self.dim_in, self.dim_in, 1))
        self.batch_target = np.zeros((self.batch_size, self.dim_in, self.dim_in, self.dim_in, self.Ncl))

        # Callbacks for Save weights and Clear Memory
        callbacks.ClearMemoryCallback()
        save_weights_callback = callbacks.SaveWeightsCallback(self.path_out)
        learning_rate_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_f1",
            factor=0.75,
            patience=6,
            min_lr=1e-6,
            verbose=1,
        )

        # Create Initial Datasets to Call During Training
        swap_callback = callbacks.DatasetSwapCallback(self, path_train, path_valid)
        (initial_train_dataset, initial_valid_dataset) = swap_callback.generate_new_tensorflow_datasets()

        plotting_callback = callbacks.TrainingPlotCallback(
            validation_data=initial_valid_dataset,
            validation_steps=self.steps_per_valid,
            path_out=self.path_out,
            label_list=self.label_list,
        )
        swap_callback.plotting_callback = plotting_callback

        # Save Training Parameters as JSON
        self.save_training_parameters(path_train, path_valid, 
                                      self.model_parameters, 
                                      learning_rate_callback)
        
        # Train the model using model.fit()
        self.display("Launch training ...")
        self.net.fit(
            initial_train_dataset,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            class_weight=self.class_weights,
            validation_data=initial_valid_dataset,
            validation_steps=self.steps_per_valid,
            callbacks=[
                tensorboard_callback,
                save_weights_callback,
                plotting_callback,
                swap_callback,
                learning_rate_callback,
            ],
            verbose=1,
        )

        self.net.save(self.path_out + "net_weights_FINAL.h5")

    def create_tf_dataset(
        self,
        copick_path,
        input_dataset,
        input_target,
        batch_size,
        dim_in,
        Ncl,
        flag_batch_bootstrap=True,
        tomoIDs=None,
        targets=None,
    ):
        copickRoot = copick.from_file(copick_path)
        organizedPicksDict = core.query_available_picks(copickRoot, tomoIDs, targets)
        dataset = tf.data.Dataset.from_generator(
            lambda: self.copick_data_generator(
                input_dataset,
                input_target,
                batch_size,
                dim_in,
                Ncl,
                flag_batch_bootstrap,
                organizedPicksDict,
            ),
            output_signature=(
                tf.TensorSpec(shape=(batch_size, dim_in, dim_in, dim_in, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(batch_size, dim_in, dim_in, dim_in, Ncl), dtype=tf.float32),
            ),
        )
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    def copick_data_generator(
        self,
        input_dataset,
        input_target,
        batch_size,
        dim_in,
        Ncl,
        flag_batch_bootstrap,
        organizedPicksDict,
    ):
        p_in = int(np.floor(dim_in / 2))

        tomodim = input_dataset[organizedPicksDict["tomoIDlist"][0]].shape
        while True:
            pool = core.get_copick_boostrap_idx(organizedPicksDict, batch_size) if flag_batch_bootstrap else None
            # pool = range(0, len(objlist))

            idx_list = []
            for i in range(batch_size):
                randomBSidx = np.random.choice(pool["bs_idx"])
                idx_list.append(randomBSidx)

                index = np.where(pool["bs_idx"] == randomBSidx)[0][0]

                x, y, z = core.get_copick_patch_position(
                    tomodim,
                    p_in,
                    self.Lrnd,
                    self.voxelSize,
                    pool["protein_coords"][index],
                )

                patch_data = input_dataset[pool["tomoID_idx"][index]][
                    z - p_in : z + p_in,
                    y - p_in : y + p_in,
                    x - p_in : x + p_in,
                ]
                patch_target = input_target[pool["tomoID_idx"][index]][
                    z - p_in : z + p_in,
                    y - p_in : y + p_in,
                    x - p_in : x + p_in,
                ]

                patch_target = to_categorical(patch_target, Ncl)
                patch_data = (patch_data - np.mean(patch_data)) / np.std(patch_data)

                # Apply Augmentations to Cropped Volume
                patch_data, patch_target = self.data_augmentor.apply_augmentations(patch_data, patch_target)

                self.batch_target[i] = patch_target
                self.batch_data[i, :, :, :, 0] = patch_data

            yield self.batch_data, self.batch_target

    def export_training_parameters(self):
        """
        Exports the training parameters as a Pydantic model.

        Returns:
            TrainingParameters: A Pydantic model containing the training parameters.
        """
        # Convert the keys of class_weights to strings
        class_weights_str_keys = {str(key): value for key, value in self.class_weights.items()}

        return settings.TrainingParameters(
            n_class = self.Ncl,
            dim_in = self.dim_in,
            batch_size=self.batch_size,
            epochs = self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            steps_per_valid=self.steps_per_valid,
            num_sub_epoch=self.NsubEpoch,
            sample_size=self.sample_size,
            loss=self.loss.name,
            class_weights=class_weights_str_keys
        )
    
    def export_lr_parameters(self, callback):
        """
        Exports the learning rate parameters from a callback as a Pydantic model.

        Args:
            callback: The learning rate scheduler callback instance, typically from Keras/TensorFlow.
        Returns:
            LearningRateParameters: A Pydantic model containing the learning rate parameters.
        """        
        return settings.LearningRateParameters(
            learning_rate=self.learning_rate, 
            min_learning_rate=callback.min_lr,
            monitor=callback.monitor,
            factor=callback.factor,
            patience=callback.patience
        )

    def save_training_parameters(self, path_train, path_valid, model_parameters, callback):
        """
        Saves the training configuration, including paths, model parameters, and training parameters, to a JSON file.

        Args:
            path_train (str): The file path to the training dataset configuration.
            path_valid (str): The file path to the validation dataset configuration.
            model_parameters (ModelParameters): A Pydantic model containing the model's architecture parameters.
        """        
        input = settings.ProcessingInput(config_path_train=path_train, 
                                         config_path_valid=path_valid)
        output = settings.ProcessingOutput(out_dir=self.path_out)
        training = self.export_training_parameters()
        learnRate = self.export_lr_parameters(callback)
        train_config = settings.ExperimentConfig(
            input=input, 
            output=output, 
            network_architecture=model_parameters, 
            training_params=training, 
            learning_params=learnRate
        )
        train_config.save_to_json()
        print('\nTraining Parameters: ', json.dumps(train_config.dict(),indent=4) )