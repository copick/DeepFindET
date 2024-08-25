from deepfindET.models.res_unet import my_res_unet_model
from deepfindET.models.unet import my_unet_model 
from deepfindET import settings
import os

def load_model(dim_in, Ncl, model_name, trained_weights_path=None, filters = [48, 64, 128], dropout_rate = 0):

    # Play with Model to Use
    assert model_name in ['unet', 'res_unet'], "Invalid model name specified. Use 'unet' or 'res_unet'."

    if model_name == 'unet':
        net = my_unet_model(dim_in, Ncl, filters, dropout_rate)
    elif model_name == 'res_unet':
        net = my_res_unet_model(dim_in, Ncl, filters, dropout_rate)
    else:
        raise ValueError("Invalid model name specified. Valid options {unet, or res_unet}")

    if trained_weights_path is not None:
        if not os.path.exists(trained_weights_path):
            raise FileNotFoundError(f"The specified path for trained weights does not exist: {trained_weights_path}")
        net.load_weights(trained_weights_path)
        print(f'\nTraining {model_name} with {trained_weights_path} Weights\n')
    else:
        print(f'\nTraining {model_name} with Randomly Initialized Weights\n')        

    # Define model parameters
    model_parameters = settings.NetworkParameters(
        architecture=model_name,
        layers=filters,
        dropout_rate=dropout_rate
    )

    return net, model_parameters
