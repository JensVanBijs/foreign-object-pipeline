#%%
import numpy as np
import os
import astra
from imageio import get_writer


def load_phantom(detector_dims, phantom_obj):
    vol_geom = astra.creators.create_vol_geom(*detector_dims)
    phantom_id = astra.data3d.create('-vol', vol_geom, data=phantom_obj)
    return vol_geom, phantom_id

def create_projections(phantom_id, vol_geom):
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
    return projections_id, projections

def add_poisson_noise(projections):
    n = projections
    n[n < 0] = 0
    projections += np.random.poisson(n)
    return projections

def project(phantom, noise=False):
    volume_geometry, phantom_id = load_phantom(detector_dims, phantom)
    projection_id, projections = create_projections(phantom_id, volume_geometry)
    if noise:
        projections = add_poisson_noise(projections)
    return projections

def create_experiment_projections(phantom_dir, noise = False, random_seed = None, outdir = "./data/object_projections"):
    if random_seed != None:
        np.random.seed(random_seed)
    proj_dir = os.path.abspath(outdir)
    if not os.path.isdir(proj_dir):
        os.mkdir(proj_dir)

    phantom_dir_list = sorted(os.listdir(os.path.abspath(phantom_dir)), key=lambda x: int(x.split("_")[0]))
    for p_dir in phantom_dir_list:
        n_phantoms = int(p_dir.split("_")[0])
        phantom_proj_dir = os.path.join(proj_dir, f"{n_phantoms}_objects")
        if not os.path.isdir(phantom_proj_dir):
            os.mkdir(phantom_proj_dir)
        for obj_idx, p_path in enumerate(os.listdir(os.path.join(phantom_dir, p_dir))):
            projection_dir = os.path.join(phantom_proj_dir, f"object_{obj_idx}")
            if not os.path.isdir(projection_dir):
                os.mkdir(projection_dir)

            phantom_path = os.path.abspath(os.path.join(phantom_dir, p_dir, p_path))
            phantom = np.load(phantom_path)
            projections = project(phantom)

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

    create_experiment_projections(phantom_dir=os.path.abspath("./data/phantoms"), noise=False)
    create_experiment_projections(phantom_dir=os.path.abspath("./data/test_phantoms"), noise=False, outdir="./data/test_object_projections")