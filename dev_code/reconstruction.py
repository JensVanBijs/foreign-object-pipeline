#%% Imports
from __future__ import division
import numpy as np
from os import mkdir
from os.path import join, isdir
from imageio import imread, imwrite
import astra
 
#%% Configuration
distance_source_origin = 44.14  # [mm]
distance_origin_detector = 69.80 - 44.14  # [mm]
detector_pixel_size = 1.05  # [mm]
detector_rows, detector_cols = 128, 128  # Size of detector [pixels].

num_of_projections = 180
angles = np.linspace(0, 2 * np.pi, num=num_of_projections, endpoint=False)

input_dir = 'projection_data'
 
#%% Load projections
projections = np.zeros((detector_rows, num_of_projections, detector_cols))
for i in range(num_of_projections):
    im = imread(join(input_dir, 'proj%04d.tif' % i)).astype(float)
    im /= 65535  # NOTE: uint16
    projections[:, i, :] = im
 
#%% Transfer projection images into ASTRA
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
 
#%% Create reconstruction
vol_geom = astra.creators.create_vol_geom(detector_cols, detector_cols, detector_rows)
reconstruction_id = astra.data3d.create('-vol', vol_geom, data=0)

# TODO: Use SIRT? (SIRT3D_CUDA)
algorithm_options = [
    "FDK_CUDA",
    "SIRT3D_CUDA"
]

for algorithm in algorithm_options:
    alg_cfg = astra.astra_dict(algorithm)
    alg_cfg['ProjectionDataId'] = projections_id
    alg_cfg['ReconstructionDataId'] = reconstruction_id

    if algorithm == "FDK_CUDA":
        output_dir = "FDK"
        alg_cfg["ShortScan"] = True
    elif algorithm == "SIRT3D_CUDA":
        output_dir = "SIRT"

    algorithm_id = astra.algorithm.create(alg_cfg)
    astra.algorithm.run(algorithm_id)
    reconstruction = astra.data3d.get(reconstruction_id)
    
    if not isdir(output_dir):
        mkdir(output_dir)
    for i in range(detector_rows):
        im = reconstruction[i, :, :]
        im = np.flipud(im)
        imwrite(join(output_dir, 'reco%04d.tiff' % i), im)
 
#%% Cleanup.
astra.algorithm.delete(algorithm_id)
astra.data3d.delete(reconstruction_id)
astra.data3d.delete(projections_id)
