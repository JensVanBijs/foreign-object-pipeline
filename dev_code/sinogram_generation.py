#%% Imports
from __future__ import division
import numpy as np
from os import mkdir
from os.path import join, isdir
from imageio import get_writer
import astra
 
#%% Configuration
distance_source_origin = 44.14  # [mm]
distance_origin_detector = 69.80 - 44.14  # [mm]
detector_pixel_size = 1.05  # [mm]
detector_rows, detector_cols = 128, 128  # Size of detector [pixels].

num_of_projections = 180
angles = np.linspace(0, 2 * np.pi, num=num_of_projections, endpoint=False)

output_dir = 'noisy_projection_data'
 
#%% Load phantom
vol_geom = astra.creators.create_vol_geom(detector_cols, detector_cols, detector_rows)
phantom = np.load("./phantom.npy")
phantom_id = astra.data3d.create('-vol', vol_geom, data=phantom)
 
#%% Create projections indexed in slices where slice 0 is the top
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

# %% Add poisson noise to the projections
m = projections
m[m < 0] = 0
projections += np.random.poisson(m)
# projections[projections > 1.1] = 1.1
# projections /= 1.1
 
#%% Save projections.
if not isdir(output_dir):
    mkdir(output_dir)
for i in range(num_of_projections):
    projection = projections[:, i, :]
    with get_writer(join(output_dir, 'proj%04d.tif' %i)) as writer:
        writer.append_data(projection)
 
#%% Cleanup.
astra.data3d.delete(projections_id)
astra.data3d.delete(phantom_id)