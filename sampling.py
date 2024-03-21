import random
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import draw
import time
from tqdm import tqdm

from utils import map2world, world2map, SDF_RT, polygon_SDF


def is_free_space(x, y, map_img, ratio):
    """ Check if a point in world coordinates is in free space on the map. """
    map_origin = np.array([[len(map_img) // 2], [len(map_img[0]) // 2], [np.pi / 2]])
    coordinates_m = world2map(map_origin, ratio, np.array([[x], [y], [0]]))
    coordinates_m = coordinates_m.astype(np.int16)
    map_x, map_y, _ = coordinates_m
    return 0 <= map_x < map_img.shape[0] and 0 <= map_y < map_img.shape[1] and map_img[map_x, map_y] == 0


def sample_free_space(scale, map_data, ratio, num_points=1):
    """ Randomly sample a point and orientation in the free space of the map.

        scale: The size of the world map is (-scale, scale)
    """
    sampled_points = []
    while len(sampled_points) < num_points:
        x, y = random.uniform(-scale, scale), random.uniform(-scale, scale)
        if is_free_space(x, y, map_data, ratio):
            orientation = random.uniform(0, 2 * np.pi)  # Orientation in radians
            sampled_points.append(np.array([x, y, orientation]))

    return sampled_points


def add_border_to_map(map_data):

    map_data[:5, :] = 1
    map_data[-5:, :] = 1
    map_data[:, :5] = 1
    map_data[:, -5:] = 1
    return map_data


def sample_random_points_around_squares(scale, map_data, ratio, num_points=1, distance_range=40):

    # Reducing the scale such that not to sample points very close to border
    scale -= 20
    sampled_points = []
    while len(sampled_points) < num_points:
        x, y = random.uniform(-scale, scale), random.uniform(-scale, scale)
        orientation = random.uniform(0, 2 * np.pi)

        map_origin = np.array([[len(map_data) // 2], [len(map_data[0]) // 2], [np.pi / 2]])
        coordinates_m = world2map(map_origin, ratio, np.array([[x], [y], [0]]))
        coordinates_m = coordinates_m.astype(np.int16)
        map_x, map_y, _ = coordinates_m
        map_x, map_y = map_x[0], map_y[0]

        if 0 <= map_x < map_data.shape[0] and 0 <= map_y < map_data.shape[1] and map_data[map_x, map_y] == 0:
            if np.any(map_data[max(0, map_x - distance_range//2):min(map_data.shape[0], map_x + distance_range//2),
                               max(0, map_y - distance_range//2):min(map_data.shape[1], map_y + distance_range//2)]):
                sampled_points.append(np.array([x, y, orientation]))

    return sampled_points


def generate_random_map(map_size):
    """Generates a random square map of a given size"""
    # Create a blank numpy map
    map_data = np.zeros((map_size, map_size)).astype(float)

    num_squares = np.random.randint(10, 20)
    # Generate random square polygons
    for _ in range(num_squares):
        # Random size and position
        size = random.randint(5, 20)
        x = random.randint(0, map_size - size)
        y = random.randint(0, map_size - size)

        # Add the square to the map
        map_data[y:y + size, x:x + size] = 1

    return add_border_to_map(map_data)


def generate_dataset(
        num_samples,
        alpha,
        r,
        dataset_name="dataset",
        cat='train',
        scale=200,
        map_size=1000,
        negative_sdf=False
):
    """
    Generate a random dataset
        scale: The size of the world map is (-scale, scale)
        negative_sdf: Only generate samples which have a negative SDF value
    """

    label = []
    robot_positions = []
    target_positions = []
    free_fov_polygon_vertices = []
    occ_fov_polygon_vertices = []
    ratio = map_size/(scale*2)

    random_map = generate_random_map(map_size)

    # Create empty map to get unoccluded FOV but bound it
    empty_map = np.zeros_like(random_map, np.uint8)
    empty_map = add_border_to_map(empty_map)

    map_folder = f'map_{time.time()}'
    os.makedirs(f'./{dataset_name}/{cat}/{map_folder}', exist_ok=True)
    plt.imsave(f'./{dataset_name}/{cat}/{map_folder}/map.png', random_map)

    free_robot_positions = sample_free_space(scale, random_map, ratio, num_samples//2)
    robot_positions_near_obstacles = sample_random_points_around_squares(scale, random_map, ratio, num_samples//2)

    for i, robot_w in enumerate(free_robot_positions + robot_positions_near_obstacles):

        map_origin = np.array([[len(random_map) // 2], [len(random_map[0]) // 2], [np.pi / 2]])
        robot_m = world2map(map_origin, ratio, robot_w.reshape((3, 1))).squeeze()
        robot_positions.append(robot_m)

        visibility_m = SDF_RT(robot_m, alpha, r, 50, random_map)
        visibility_w = (map2world(map_origin, ratio, visibility_m.T)[0:2, :]).T
        occ_fov_polygon_vertices.append(visibility_m)

        # Generate visible region image (implement based on your camera model)
        visibility_free = SDF_RT(robot_m, alpha, r, 50, empty_map)
        free_fov_polygon_vertices.append(visibility_free)

        # Create input images
        occ_fov_img, free_fov_img = np.zeros_like(random_map, np.uint8), np.zeros_like(random_map, np.uint8)

        rr, cc = draw.polygon(visibility_free[:, 0], visibility_free[:, 1])
        free_fov_img[rr, cc] = 1

        rr, cc = draw.polygon(visibility_m[:, 0], visibility_m[:, 1])
        occ_fov_img[rr, cc] = 1

        # Generate target position
        if negative_sdf:
            # Find points inside the fov
            points_inside_fov = np.stack(np.where(occ_fov_img==1))
            # Sample a random point
            s = np.random.choice(points_inside_fov.shape[1])
            target_m = points_inside_fov[:, s]
            target_w = map2world(map_origin, ratio, target_m.reshape((-1,1))).squeeze()
        else:
            target_w = sample_free_space(scale, random_map, ratio)[0]
            target_m = world2map(map_origin, ratio, target_w.reshape((3, 1))).squeeze()
        target_positions.append(target_m)
        
        signed_distance = polygon_SDF(visibility_w, target_w[0:2])


        os.makedirs(f'./{dataset_name}/{cat}/{map_folder}/free_fov_img', exist_ok=True)
        os.makedirs(f'./{dataset_name}/{cat}/{map_folder}/occ_fov_img', exist_ok=True)

        plt.imsave(f'./{dataset_name}/{cat}/{map_folder}/free_fov_img/{i:05d}.png', free_fov_img)
        plt.imsave(f'./{dataset_name}/{cat}/{map_folder}/occ_fov_img/{i:05d}.png', occ_fov_img)
        label.append(signed_distance * ratio)

    np.savez_compressed(
        f'./{dataset_name}/{cat}/{map_folder}/label',
        signed_distance=label,
        robot_positions=robot_positions,
        target_positions=target_positions
    )

    np.save(f'./{dataset_name}/{cat}/{map_folder}/free_fov_polygon_vertices.npy', np.array(free_fov_polygon_vertices, dtype=object),
            allow_pickle=True)
    np.save(f'./{dataset_name}/{cat}/{map_folder}/occ_fov_polygon_vertices.npy', np.array(occ_fov_polygon_vertices, dtype=object),
            allow_pickle=True)

    return True


# Example usage
if __name__ == "__main__":

    num_maps = 1000
    print("Generating Training Data")
    for _ in tqdm(range( int(0.8 * num_maps))):
      _ = generate_dataset(
          20,
          np.pi / 3,
          100,
          dataset_name="dataset",
          cat='train',
          negative_sdf=True
      )

    print("Generating Test Data")
    for _ in tqdm(range(int(0.2 * num_maps))):
        _ = generate_dataset(
            20,
            np.pi / 3,
            100,
            dataset_name="dataset",
            cat='test',
            negative_sdf=True
        )
