import deepfindET.utils.copick_tools as tools
from deepfindET.inference import Segment
import tensorflow as tf
import click, copick

@click.group()
@click.pass_context
def cli(ctx):
    pass

@cli.command()
@click.option(
    "--predict-config",
    type=str,
    required=True,
    help="Path to the copick config file.",
)
@click.option(
    "--model-name",
    type=str,
    required=False,
    default="res_unet",
    show_default=True,
    help="Model Architecture Name to Load For Inference",
)
@click.option(
    "--path-weights",
    type=str,
    required=True,
    help="Path to the Trained Model Weights.",
)
@click.option(
    "--n-class",
    type=int,
    required=True,
    help="Number of classes.",
)
@click.option(
    "--patch-size",
    type=int,
    required=True,
    help="Patch of Volume for Input to Network.",
)
@click.option(
    "--user-id", 
    type=str, 
    default="deepfindET", 
    show_default=True, 
    required=False,
    help="User ID filter for input."
)
@click.option(
    "--session-id", 
    type=str, 
    default=None, 
    show_default=True, 
    required=True,
    help="Session ID filter for input."
)
@click.option(
    "--voxel-size",
    type=float,
    required=False,
    default=10.0,
    show_default=True,
    help="Voxel size of the tomograms to segment.",
)
@click.option(
    "--tomogram-algorithm",
    type=str,
    required=False,
    default="wbp",
    show_default=True,
    help="Tomogram Algorithm.",
)
@click.option(
    "--parallel-mpi/--no-parallel-mpi",
    default=False,
    required=False,
    show_default=True,
    help="Patch of Volume for Input to Network.",
)
@click.option(
    "--tomo-ids",
    type=str,
    required=False,
    default=None,
    show_default=True,
    help="Tomogram IDs to Segment.",
)
@click.option(
    "--output-scoremap",
    type=bool,
    required=False,
    default=False,
    show_default=True,
    help="Output scoremap.",
)
@click.option(
    "--scoremap-name",
    type=str,
    required=False,
    default="scoremap",
    show_default=True,
    help="Output name for scoremap.",
)
@click.option(
    "--segmentation-name",
    type=str,
    required=False,
    default="segmentation",
    show_default=True,
    help="Output name for segmentation.",
)
def segment(
    predict_config: str,
    model_name: str,
    path_weights: str,
    n_class: int,
    patch_size: int,
    user_id: str,
    session_id: str,
    voxel_size: float = 10,
    tomogram_algorithm: str = "denoised",
    parallel_mpi: bool = False,
    tomo_ids: str = None,
    output_scoremap: bool = False,
    scoremap_name: str = "scoremap",
    segmentation_name: str = "segmentation",
):

    inference_tomogram_segmentation(predict_config, model_name, path_weights, n_class, patch_size, 
                                    user_id, session_id, voxel_size, tomogram_algorithm, parallel_mpi, tomo_ids,
                                    output_scoremap, scoremap_name, segmentation_name)

def inference_tomogram_segmentation(
    predict_config: str,
    model_name: str,
    path_weights: str,
    n_class: int,
    patch_size: int,
    user_id: str,
    session_id: str,
    voxel_size: float = 10,
    tomogram_algorithm: str = "denoised",
    parallel_mpi: bool = False,
    tomo_ids: str = None,
    output_scoremap: bool = False,
    scoremap_name: str = "scoremap",
    segmentation_name: str = "segmentation",        
    ):

    # Determine if Using MPI or Sequential Processing
    if parallel_mpi:
        from mpi4py import MPI

        # Initialize MPI (Get Rank and nProc)
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nProcess = comm.Get_size()

        locGPU = rank % len(tf.config.list_physical_devices('GPU'))
    else:
        nProcess = 1
        rank = 0
        locGPU = None

    ############## (Step 1) Initialize segmentation task: ##############

    # Load CoPick root
    copickRoot = copick.from_file(predict_config)

    seg = Segment(n_class, model_name, path_weights=path_weights, patch_size=patch_size, gpuID = locGPU)

    # Load Evaluate TomoIDs
    evalTomos = tomo_ids.split(",") if tomo_ids is not None else [run.name for run in copickRoot.runs]

    # Create Temporary Empty Folder
    for tomoInd in range(len(evalTomos)):
        if (tomoInd + 1) % nProcess == rank:
            # Extract TomoID and Associated Run
            tomoID = evalTomos[tomoInd]
            print(f'Processing Run: {tomoID}')

            # Load data:
            tomo = tools.get_copick_tomogram(
                copickRoot,
                voxelSize=voxel_size,
                tomoAlgorithm=tomogram_algorithm,
                tomoID=tomoID,
            )

            # Segment tomogram:
            scoremaps = seg.launch(tomo[:])

            # Query Copick Runs
            copickRun = copickRoot.get_run(tomoID)

            # Write scoremaps to file:
            if output_scoremap:
                tools.write_ome_zarr_scoremap(
                    copickRun,
                    scoremaps,
                    voxelSize=voxel_size,
                    scoremapName=scoremap_name,
                    userID=user_id,
                    sessionID=session_id,
                    tomo_type=tomogram_algorithm,
                )

            # Get labelmap from scoremaps:
            labelmap = seg.to_labelmap(scoremaps)
            tools.write_ome_zarr_segmentation(
                copickRun,
                labelmap,
                voxelSize=voxel_size,
                segmentationName=segmentation_name,
                userID=user_id,
                sessionID=session_id,
            )

    print("Segmentations Complete!")

if __name__ == "__main__":
    cli()