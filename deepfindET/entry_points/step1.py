from typing import List, Tuple, Union
import click, copick, zarr
import numpy as np

import deepfindET.utils.copick_tools as tools
from deepfindET.utils.target_build import TargetBuilder

@click.group()
@click.pass_context
def cli(ctx):
    pass

def parse_target(ctx, param, value):
    targets = []
    for v in value:
        parts = v.split(',')
        if len(parts) == 2:
            obj_name, radius = parts
            targets.append((obj_name, None, None, int(radius)))
        elif len(parts) == 4:
            obj_name, user_id, session_id, radius = parts
            targets.append((obj_name, user_id, session_id, int(radius)))
        else:
            raise click.BadParameter('Each target must be in the form "name,radius" or "name,user_id,session_id,radius"')
    return targets

def parse_seg_target(ctx, param, value):
    seg_targets = []
    for v in value:
        parts = v.split(',')
        if len(parts) == 1:
            name = parts[0]
            seg_targets.append((name, None, None))
        elif len(parts) == 3:
            name, user_id, session_id = parts
            seg_targets.append((name, user_id, session_id))
        else:
            raise click.BadParameter('Each seg-target must be in the form "name" or "name,user_id,session_id"')
    return seg_targets    

@cli.command(context_settings={"show_default": True})
@click.option("--config", type=str, required=True, help="Path to the configuration file.")
@click.option(
    "--target",
    multiple=True,
    required=False,
    callback=parse_target,
    help='Target specification as "name,radius" or "name,user_id,session_id,radius".',
)
@click.option(
    "--seg-target",
    multiple=True,
    required=False,
    callback=parse_seg_target,
    help='Segmentation target specification as "name" or "name,user_id,session_id".',
)
@click.option(
    "--tomo-ids",
    type=str,
    required=False,
    default=None,
    show_default=True,
    help="Comma separated list of Tomogram IDs.",
)
@click.option(
    "--voxel-size", 
    type=float, 
    default=10, 
    help="Voxel size.",
)
@click.option("--tomogram-algorithm", type=str, default="wbp", help="Tomogram algorithm.")
@click.option("--out-name", type=str, default="spheretargets", help="Target name.")
@click.option("--out-user-id", type=str, default="train-deepfinder", help="User ID for output.")
@click.option("--out-session-id", type=str, default="0", help="Session ID for output.")
def create(
    config: str,
    target: List[Tuple[str, Union[str, None], Union[str, None], int]],
    seg_target: List[Tuple[str, Union[str, None], Union[str, None]]],
    tomo_ids: str,
    voxel_size: float,
    tomogram_algorithm: str = "wbp",
    out_name: str = "spheretargets",
    out_user_id: str = "train-deepfinder",
    out_session_id: str = "0",
    ):

    # Load CoPick root
    copickRoot = copick.from_file(config)

    train_targets = {}
    for t in target:
        obj_name, user_id, session_id, radius = t
        info = {
            "label": copickRoot.get_object(obj_name).label,
            "user_id": user_id,
            "session_id": session_id,
            "radius": radius,
            "is_particle_target": True,
        }
        train_targets[obj_name] = info

    for s in seg_target:
        obj_name, user_id, session_id = s
        info = {
            "label": copickRoot.get_object(obj_name).label,
            "user_id": user_id,
            "session_id": session_id,
            "radius": None,       
            "is_particle_target": False,                 
        }
        train_targets[obj_name] = info

    # Additional logic to source data from the data portal can be added here
    create_train_targets(config, train_targets, tomo_ids, voxel_size, tomogram_algorithm, out_name, out_user_id, out_session_id)

def create_train_targets(
    config: str,
    train_targets: dict,
    tomo_ids: str,
    voxel_size: float,
    tomogram_algorithm: str,
    out_name: str,
    out_user_id: str,
    out_session_id: str,
    ):

    # Load CoPick root
    copickRoot = copick.from_file(config)

    target_names = list(train_targets.keys())

    # Radius list
    max_target = max(e["label"] for e in train_targets.values())
    radius_list = np.zeros((max_target,), dtype=np.uint8)

    for _key, value in train_targets.items():
        radius_list[value["label"] - 1] = value["radius"] if value["radius"] is not None else 0

    # Load tomo_ids
    tomo_ids = [run.name for run in copickRoot.runs] if tomo_ids is None else tomo_ids.split(",")

    # Add Spherical Targets to Mebranes
    targetbuild = TargetBuilder()

    # Create Empty Target Volume
    target_vol = tools.get_target_empty_tomogram(copickRoot, voxelSize=voxel_size, tomoAlgorithm=tomogram_algorithm)

    # Iterate Through All Runs
    for tomoID in tomo_ids:
        
        # Extract TomoID and Associated Run
        print(f'Processing Run: {tomoID}')
        copickRun = copickRoot.get_run(tomoID)

        # Reset Target As Empty Array
        # (If Membranes or Organelle Segmentations are Available, add that As Well)
        target_vol[:] = 0

        # Applicable segmentations
        query_seg = []
        for target_name in target_names:
            if not train_targets[target_name]["is_particle_target"]:            
                query_seg += copickRun.get_segmentations(
                    name=target_name,
                    user_id=train_targets[target_name]["user_id"],
                    session_id=train_targets[target_name]["session_id"],
                    voxel_size=voxel_size,
                    is_multilabel=False,
                )                

        # Add Segmentations to Target
        for seg in query_seg:
            classLabel = copickRoot.get_object(seg.name).label
            segvol = zarr.open(seg.zarr())["0"]

            target_vol[:] = np.array(segvol) * classLabel

        # Applicable picks
        query = []
        for target_name in target_names:
            if train_targets[target_name]["is_particle_target"]:
                query += copickRun.get_picks(
                    object_name=target_name,
                    user_id=train_targets[target_name]["user_id"],
                    session_id=train_targets[target_name]["session_id"],
                )
        
        # Read Particle Coordinates and Write as Segmentation
        objl_coords = []
        for picks in query:
            classLabel = copickRoot.get_object(picks.pickable_object_name).label

            if classLabel is None:
                print("Missing Protein Label: ", picks.pickable_object_name)
                exit(-1)

            for ii in range(len(picks.points)):
                objl_coords.append(
                    {
                        "label": classLabel,
                        "x": picks.points[ii].location.x / voxel_size,
                        "y": picks.points[ii].location.y / voxel_size,
                        "z": picks.points[ii].location.z / voxel_size,
                        "phi": 0,
                        "psi": 0,
                        "the": 0,
                    },
                )

        # Create Target For the Given Coordinates and Sphere Diameters
        target = targetbuild.generate_with_spheres(objl_coords, target_vol, radius_list).astype(np.uint8)

        # Write the Target Tomogram as OME Zarr
        tools.write_ome_zarr_segmentation(copickRun, target, voxel_size, out_name, out_user_id, out_session_id)

if __name__ == "__main__":
    cli()
