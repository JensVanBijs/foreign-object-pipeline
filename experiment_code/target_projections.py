#%%
import numpy as np
import astra
import diplib as dip
import os
from imageio import get_writer

def read_reconstruction(reconstruction_path):
    reconstruction = np.load(reconstruction_path)
    return reconstruction

def segment(reconstruction):
    recon = dip.Image(reconstruction)
    object_mask = np.asarray(dip.OtsuThreshold(recon))
    
    object = reconstruction
    object[~object_mask] = np.mean(reconstruction[object_mask])
    object = dip.Image(object)

    ground_truth = np.asarray(dip.OtsuThreshold(object))
    return ground_truth

def create_projections(ground_truth):
    # Create the volume geometry
    vol_geom = astra.creators.create_vol_geom(detector_cols, detector_cols, detector_rows)

    # Combine phantom and volume geometry into a data object
    phantom_id = astra.data3d.create('-vol', vol_geom, data=ground_truth)

    proj_geom = astra.create_proj_geom(
            'cone',
            1,
            1,
            detector_rows,
            detector_cols,
            angles,
            (distance_source_origin + distance_origin_detector) / detector_pixel_size,
            0
    )

    projections_id, projections = astra.creators.create_sino3d_gpu(phantom_id, proj_geom, vol_geom)
    return projections

def project(reconstruction_path):
    reconstruction = read_reconstruction(reconstruction_path)
    ground_truth = segment(reconstruction)
    projections = create_projections(ground_truth)
    return projections

def create_experiment_target_projections(reconstruction_dir, outdir = "./data/target_projections"):
    target_dir = os.path.abspath(outdir)
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    recon_dir_list = sorted(os.listdir(os.path.abspath(reconstruction_dir)), key=lambda x: int(x.split("_")[0]))
    for r_dir in recon_dir_list:
        n_recons = int(r_dir.split("_")[0])
        recon_proj_dir = os.path.join(target_dir, f"{n_recons}_objects")
        if not os.path.isdir(recon_proj_dir):
            os.mkdir(recon_proj_dir)
        for obj_idx, r_path in enumerate(os.listdir(os.path.join(reconstruction_dir, r_dir))):
            projection_dir = os.path.join(recon_proj_dir, f"targets_{obj_idx}")
            if not os.path.isdir(projection_dir):
                os.mkdir(projection_dir)

            recon_path = os.path.abspath(os.path.join(reconstruction_dir, r_dir, r_path))
            projections = project(recon_path)

            for proj_idx in range(n_projections):
                projection = projections[:, proj_idx, :]
                with get_writer(os.path.join(projection_dir, f"projection_{proj_idx}.tif")) as writer:
                    writer.append_data(projection)

if __name__ == "__main__":
    distance_source_origin = 44.14  # [mm]
    distance_origin_detector = 69.80 - 44.14  # [mm]
    detector_pixel_size = 1.05  # [mm]
    detector_rows, detector_cols = 128, 128  # [pixels]
    detector_dims = (detector_cols, detector_cols, detector_rows)

    n_projections = 180
    angles = np.linspace(0, 2 * np.pi, num=n_projections, endpoint=False)

    create_experiment_target_projections(reconstruction_dir="./data/object_reconstructions")
    create_experiment_target_projections(reconstruction_dir="./data/test_object_reconstructions", outdir="./data/test_target_projections")