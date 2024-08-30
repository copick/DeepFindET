from tensorflow.keras.optimizers import Adam
from deepfindET.utils import copick_tools as copicktools
from deepfindET.utils import core

from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
import numpy as np
import os

class DatasetSwapCallback(tf.keras.callbacks.Callback):
    def __init__(self, train_instance, path_train, path_valid, plotting_callback=None):
        """
        Callback to swap datasets every N epochs during training.

        Args:
            model_instance: The instance of the model being trained.
            batch_size (int): The batch size for training.
            dim_in (int): The dimension of the input data.
            Ncl (int): The number of classes.
            NsubEpoch (int): The number of epochs after which to swap datasets.
            sample_size (int): The number of IDs to sample for each swap.
            trainTomoIDs (list): The list of all possible training tomogram IDs.
            validTomoIDs (list): The list of all possible validation tomogram IDs.
            path_train (str): The path to the training data.
            path_valid (str, optional): The path to the validation data. If None, trainTomoIDs and validTomoIDs must be provided. Default is None.
            flag_batch_bootstrap (bool, optional): Whether to use batch bootstrap sampling. Default is True.
        """

        super().__init__()
        self.train_instance = train_instance
        self.epoch_count = 0
        self.plotting_callback = plotting_callback
        self.path_train = path_train
        self.path_valid = path_valid

    def generate_new_tensorflow_datasets(self):
        # Validate that either both training and validation paths are available, or both sets of tomoIDs are provided
        if self.path_valid is None and (
            self.train_instance.trainTomoIDs is None or self.train_instance.validTomoIDs is None
        ):
            raise ValueError(
                "Either 'path_valid' must be provided or both 'self.trainTomoIDs' and 'self.validTomoIDs' must be set.",
            )

        # Extract RunIDs for Validation and Training Datasets
        if self.path_valid is not None:
            self.train_instance.trainTomoIDs = copicktools.get_copick_project_tomoIDs(self.path_train)
            self.train_instance.validTomoIDs = copicktools.get_copick_project_tomoIDs(self.path_valid)
            path_valid = self.path_valid
        else:
            path_valid = self.path_train

        # Sample new sets of IDs for training and validation
        sampled_train_ids = np.random.choice(
            self.train_instance.trainTomoIDs,
            self.train_instance.sample_size,
            replace=False,
        )
        if len(self.train_instance.validTomoIDs) < self.train_instance.sample_size:
            sampled_valid_ids = self.train_instance.validTomoIDs
        else:
            sampled_valid_ids = np.random.choice(
                self.train_instance.validTomoIDs,
                self.train_instance.sample_size,
                replace=False,
            )

        # Load datasets based on the provided paths and sampled IDs
        (trainData, trainTarget) = core.load_copick_datasets(self.path_train, self.train_instance, sampled_train_ids)
        (validData, validTarget) = core.load_copick_datasets(path_valid, self.train_instance, sampled_valid_ids)

        # Create TensorFlow datasets
        train_dataset = self.train_instance.create_tf_dataset(
            self.path_train,
            trainData,
            trainTarget,
            self.train_instance.batch_size,
            self.train_instance.dim_in,
            self.train_instance.Ncl,
            self.train_instance.flag_batch_bootstrap,
            sampled_train_ids,
            self.train_instance.targets,
        )
        valid_dataset = self.train_instance.create_tf_dataset(
            path_valid,
            validData,
            validTarget,
            self.train_instance.batch_size,
            self.train_instance.dim_in,
            self.train_instance.Ncl,
            self.train_instance.flag_batch_bootstrap,
            sampled_valid_ids,
            self.train_instance.targets,
        )

        return (train_dataset, valid_dataset)

    def on_epoch_end(self, epoch, logs=None):
        """
        Swaps the datasets every NsubEpochs - called at the end of every epoch.

        Args:
            epoch (int): The index of the epoch.
            logs (dict, optional): Additional information on training progress. Default is None.
        """
        self.epoch_count += 1
        if self.epoch_count % self.train_instance.NsubEpoch == 0:
            print(f"Swapping datasets at epoch {self.epoch_count}")

            (new_train_dataset, new_valid_dataset) = self.generate_new_tensorflow_datasets()

            # Update the model's dataset
            self.train_instance.net.train_dataset = new_train_dataset
            self.train_instance.net.valid_dataset = new_valid_dataset

            if self.plotting_callback:
                self.plotting_callback.validation_data = new_valid_dataset

            # Reset the Optimizer ( TODO? )
            # new_optimizer = Adam(learning_rate=self.learning_rate, 
            #                      beta_1=self.beta1, beta_2=self.beta2, 
            #                      epsilon=self.epislon, decay=self.decay)
            # self.model.compile(optimizer=new_optimizer, 
            #                    loss=self.model.loss, 
            #                    metrics=self.model.metrics)


#  Custom callback to save model weights every 10 epochs.
class SaveWeightsCallback(tf.keras.callbacks.Callback):
    def __init__(self, path_out):
        super().__init__()
        self.path_out = path_out

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            self.model.save( os.path.join(self.path_out, f"net_weights_epoch{epoch + 1}.h5") )


# Clears the Keras backend session to free up memory - called at the end of every epoch.
class ClearMemoryCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        tf.keras.backend.clear_session()
        gc.collect()


# Track and plot training metrics at the end of each epoch.
class TrainingPlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, validation_steps, path_out, label_list):
        """
        Args:
            validation_data (tf.data.Dataset): The validation dataset.
            validation_steps (int): The number of validation steps to run.
            path_out (str): The directory path where the history and plots should be saved.
            label_list (list): The list of class labels.
        """
        super().__init__()
        self.validation_data = validation_data
        self.validation_steps = validation_steps
        self.path_out = path_out
        self.label_list = label_list
        self.history = {
            "loss": [],
            "acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "val_recall": [],
            "val_precision": [],
        }

    def on_epoch_end(self, epoch, logs=None):
        """
        Tracks and plots training metrics at the end of each Epoch.

        Args:
            epoch (int): The index of the epoch.
            logs (dict, optional): Additional information on training progress. Default is None.
        """
        # Append training metrics
        self.history["loss"].append(logs.get("loss"))
        self.history["acc"].append(logs.get("accuracy"))
        self.history["val_loss"].append(logs.get("val_loss"))
        self.history["val_acc"].append(logs.get("val_accuracy"))

        # Retrieve validation data and predictions
        val_data, val_target = [], []
        for step, (x, y) in enumerate(self.validation_data):
            if step >= self.validation_steps:
                break
            val_data.append(x.numpy())
            val_target.append(y.numpy())

        val_data = np.concatenate(val_data, axis=0)
        val_target = np.concatenate(val_target, axis=0)

        val_predict = np.argmax(self.model.predict(val_data), axis=-1)
        val_targ = np.argmax(val_target, axis=-1)

        # Calculate precision, recall, and F1-score
        scores = precision_recall_fscore_support(
            val_targ.flatten(),
            val_predict.flatten(),
            average=None,
            labels=self.label_list,
        )

        self.history["val_f1"].append(scores[2])
        self.history["val_recall"].append(scores[1])
        self.history["val_precision"].append(scores[0])

        # Save history and plot
        core.save_history(self.history, os.path.join(self.path_out, "net_train_history.h5") )
        core.plot_history(self.history, os.path.join(self.path_out, "net_train_history_plot.png") )


def log_images_func(model, validation_data, steps):
    val_data, val_target = [], []
    for step, (x, y) in enumerate(validation_data):
        if step >= steps:
            break
        val_data.append(x.numpy())
        val_target.append(y.numpy())

    val_data = np.concatenate(val_data, axis=0)
    val_target = np.concatenate(val_target, axis=0)

    val_predict = model.predict(val_data)

    return val_data, val_predict


class ImageLogger(tf.keras.callbacks.Callback):
    def __init__(self, writer, log_images_func, validation_data, steps, prefix="train"):
        super().__init__()
        self.writer = writer
        self.log_images_func = log_images_func
        self.validation_data = validation_data
        self.steps = steps
        self.prefix = prefix

    def extract_vol_infrance(self, inVol):
        return np.expand_dims(np.expand_dims(np.array(inVol), axis=0), axis=-1)

    def log_images(self, writer, images, predictions, step, prefix="train"):
        with writer.as_default():
            for i, (image, prediction) in enumerate(zip(images, predictions)):
                img_tensor = self.extract_vol_infrance(image)
                pred_tensor = self.extract_vol_infrance(prediction)

                # Ensure tensors are not None
                if img_tensor is not None and pred_tensor is not None:  # noqa: SIM102
                    # Check the rank of the tensor
                    if len(img_tensor.shape) == 4:
                        # Select a middle slice for visualization
                        img_tensor = img_tensor[img_tensor.shape[0] // 2, :, :]
                        pred_tensor = pred_tensor[pred_tensor.shape[0] // 2, :, :]

                        # Add batch and channel dimensions
                        img_tensor = np.expand_dims(np.expand_dims(img_tensor, axis=0), axis=-1)
                        pred_tensor = np.expand_dims(np.expand_dims(pred_tensor, axis=0), axis=-1)

                        tf.summary.image(f"{prefix}_image_{i}", img_tensor, step=step)
                        tf.summary.image(f"{prefix}_prediction_{i}", pred_tensor, step=step)
