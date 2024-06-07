#%% 
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools

def sphere(shape, radius, position):
    """Generate an n-dimensional spherical object."""

    assert len(position) == len(shape), "Position should be a position in as much dimensions as the shape"
    semisizes = (radius,) * len(shape)

    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]

    arr = np.zeros(shape, dtype=float)
    for x_i, semisize in zip(position, semisizes):
        arr += (x_i / semisize) ** 2

    return arr <= 1.0

def get_plane_equation_from_points(p1, p2, p3): 
    """Calculate the parameters of the hyperplane formula based on three 3D points""" 
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    x3, y3, z3 = p3
    a1 = x2 - x1 
    b1 = y2 - y1 
    c1 = z2 - z1 
    a2 = x3 - x1 
    b2 = y3 - y1 
    c2 = z3 - z1 
    a = b1 * c2 - b2 * c1 
    b = a2 * c1 - a1 * c2 
    c = a1 * b2 - b1 * a2 
    d = (- a * x1 - b * y1 - c * z1) 
    return [a, b, c, d]

def init_cube():
    # Create the coordinate space
    x, y, z = np.indices(space_shape)

    # Create the bounds for the cube position
    lower_bound = cube_shape[0] // 2
    upper_bound = space_shape[0] - lower_bound

    # Define the cube
    cube = (x > lower_bound) & (x < upper_bound) & (y > lower_bound) \
        & (y < upper_bound) & (z > lower_bound) & (z < upper_bound)
    
    return cube, lower_bound, upper_bound

def cutoff_cube(cube, lower_bound, upper_bound):
    # Get cube coordinates
    cube_locations = np.where(cube)
    cube_coords = [(cube_locations[0][i], cube_locations[1][i], cube_locations[2][i]) \
                    for i in range(len(cube_locations[0]))]

    # Calculate the line midpoint and the corner locations
    line_midpoint = np.average(cube_locations)
    cube_corners = list(itertools.product(*zip([lower_bound + 1, lower_bound + 1, lower_bound + 1], \
                                            [upper_bound - 1, upper_bound - 1, upper_bound - 1])))

    # Cutoff for each corner
    for corner in cube_corners:
        cutoff_size = np.random.choice(32, 3)
        corner_x, corner_y, corner_z = corner

        if corner_x < line_midpoint:
            x_point = [corner_x + cutoff_size[0], corner_y, corner_z]
        else:
            x_point = [corner_x - cutoff_size[0], corner_y, corner_z]
        if corner_y < line_midpoint:
            y_point = [corner_x, corner_y + cutoff_size[1], corner_z]
        else:
            y_point = [corner_x, corner_y - cutoff_size[1], corner_z]
        if corner_z < line_midpoint:
            z_point = [corner_x, corner_y, corner_z + cutoff_size[2]]
        else:
            z_point = [corner_x, corner_y, corner_z - cutoff_size[2]]

        # Calculate hyperplane formula parameters
        plane_params = get_plane_equation_from_points(x_point, y_point, z_point)
        
        # Calculate the side of the hyperplane that needs to be preserved
        sign = np.sign(np.dot(plane_params, np.array([*space_shape, 2])//2))
        
        # Remove all points on the wrong side of the hyperplane
        for x_coord, y_coord, z_coord in cube_coords:
            if np.sign(np.dot(plane_params, np.array([x_coord, y_coord, z_coord, 1]))) != sign:
                cube[x_coord][y_coord][z_coord] = False

    return cube

def insert_foreigns(cube):
    # Randomly choose to add 1 or 2 foreign objects
    n_objects = np.random.choice([1, 2], p=[.5, .5])

    for _ in range(n_objects):
        # Randomly choose the size and location of the object
        radius = np.random.choice(list(range(3, 8)))
        cube_locations = np.where(cube)
        cube_coords = [(cube_locations[0][i], cube_locations[1][i], cube_locations[2][i]) \
                    for i in range(len(cube_locations[0]))]

        random_center_idx = np.random.choice(len(cube_coords))
        object_center = cube_coords[random_center_idx]

        # Add a spherical foreign object of the chosen size at the chosen location
        foreign_object = sphere(space_shape, radius, object_center)
        cube = cube.astype(int) + foreign_object.astype(int) * 10

    return cube

def plot_space(cube):
    """Plot the 3D Cube with the foreign object(s)"""
    colours = np.empty(cube.shape)
    voxelarray = (cube > 0) | (cube < 0)

    colours = np.empty(voxelarray.shape, dtype=object)
    colours[cube < 0] = 'white'
    colours[cube > 0] = 'blue'
    colours[cube > 1] = 'red'

    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(cube.astype(float), facecolors=colours)
    plt.show()

def create_phantom(plot=False):
    phantom_obj, lower, upper = init_cube()
    phantom_obj = cutoff_cube(phantom_obj, lower, upper)
    phantom_obj = insert_foreigns(phantom_obj)
    if plot:
        plot_space(phantom_obj)
    return phantom_obj

def create_experiment_phantoms(phantom_range = 100, random_seed = None, outdir = "./data/phantoms"):
    if random_seed != None:
        np.random.seed(random_seed)
    phantom_dir = os.path.abspath(outdir)
    if not os.path.isdir(phantom_dir):
        os.mkdir(phantom_dir)

    for n in range(1, phantom_range+1):
        output_directory = os.path.join(phantom_dir, f"{n}_phantoms")
        if not os.path.isdir(output_directory):
            os.mkdir(output_directory)

        for p in range(n):
            phantom = create_phantom()
            phantom_path = os.path.join(output_directory, f"phantom_{p}.npy")
            np.save(phantom_path, phantom)

def create_test_phantoms(n_phantoms = 25, random_seed = None, outdir = "./data/test_phantoms"):
    if random_seed != None:
        np.random.seed(random_seed)
    phantom_dir = os.path.abspath(outdir)
    if not os.path.isdir(phantom_dir):
        os.mkdir(phantom_dir)
    if not os.path.isdir(os.path.join(phantom_dir, f"{n_phantoms}_phantoms")):
        os.mkdir(os.path.join(phantom_dir, f"{n_phantoms}_phantoms"))

        for p in range(n_phantoms):
            phantom = create_phantom()
            phantom_path = os.path.join(phantom_dir, f"{n_phantoms}_phantoms", f"phantom_{p}.npy")
            np.save(phantom_path, phantom)

if __name__ == "__main__":
    space_shape = (128, 128, 128)
    cube_shape = (64, 64, 64)
    create_experiment_phantoms(100)
    create_test_phantoms(random_seed=42)