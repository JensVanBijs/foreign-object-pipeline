#%%
import numpy as np
import astra
from imageio import imread, imwrite
import os

def read_projections(path_to_imgstack):
    projections = np.zeros((detector_rows, n_projections, detector_cols))
    for idx in range(n_projections):
        im = imread(os.path.join(path_to_imgstack, f"projection_{idx}.tif")).astype(float)
        im /= 65535  # NOTE: uint16
        projections[:, idx, :] = im
    return projections

def load_projections(projections):
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
    projections_id = astra.data3d.create('-sino', proj_geom, projections)
    return proj_geom, projections_id

def create_reconstruction(projections_id):
    vol_geom = astra.creators.create_vol_geom(detector_cols, detector_cols, detector_rows)
    reconstruction_id = astra.data3d.create('-vol', vol_geom, data=0)

    algorithm = "SIRT3D_CUDA"
    alg_cfg = astra.astra_dict(algorithm)

    alg_cfg['ProjectionDataId'] = projections_id
    alg_cfg['ReconstructionDataId'] = reconstruction_id

    algorithm_id = astra.algorithm.create(alg_cfg)
    astra.algorithm.run(algorithm_id, iterations=100)
    reconstruction = astra.data3d.get(reconstruction_id)

    astra.algorithm.delete(algorithm_id)
    astra.data3d.delete(reconstruction_id)
    astra.data3d.delete(projections_id)
    
    return reconstruction

def reconstruct(path_to_projections):
    projections = read_projections(path_to_projections)
    projection_geometry, projection_id = load_projections(projections)
    reconstruction = create_reconstruction(projection_id)
    return reconstruction


def create_experiment_reconstructions(projection_dir, outdir = os.path.abspath("./data/object_reconstructions")):
    reconstruction_dir = os.path.abspath(outdir)
    if not os.path.isdir(reconstruction_dir):
        os.mkdir(reconstruction_dir)

    projection_dir_list = sorted(os.listdir(os.path.abspath(projection_dir)), key=lambda x: int(x.split("_")[0]))
    for p_dir in projection_dir_list:
        n_objects = int(p_dir.split("_")[0])
        rec_dir = os.path.join(reconstruction_dir, f"{n_objects}_objects")
        if not os.path.isdir(rec_dir):
            os.mkdir(rec_dir)

        object_dir_path = os.path.join(projection_dir, p_dir)
        object_list = sorted(os.listdir(object_dir_path), key=lambda x: int(x.split("_")[-1]))
        for object_idx, object_stack in enumerate(object_list):
            object_stack_path = os.path.join(object_dir_path, object_stack)
            reconstruction_save_path = os.path.join(rec_dir, f"reconstruction_{object_idx}.npy")
            reconstruction = reconstruct(object_stack_path)
            np.save(reconstruction_save_path, reconstruction)

if __name__ == "__main__":
    distance_source_origin = 44.14  # [mm]
    distance_origin_detector = 69.80 - 44.14  # [mm]
    detector_pixel_size = 1.05  # [mm]
    detector_rows, detector_cols = 128, 128  # [pixels]
    detector_dims = (detector_cols, detector_cols, detector_rows)

    n_projections = 180
    angles = np.linspace(0, 2 * np.pi, num=n_projections, endpoint=False)

    create_experiment_reconstructions(projection_dir=os.path.abspath("./data/object_projections"))
    create_experiment_reconstructions(projection_dir=os.path.abspath("./data/test_object_projections"), outdir="./data/test_object_reconstructions")