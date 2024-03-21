import numpy as np

global_map = np.zeros((100, 100))


def map2world(map_origin, ratio, x_m):
    if len(x_m) == 2:
        x_m = np.array([[1, 0], [0, 1], [0, 0]]) @ x_m
    m2w = np.array([[0, 1 / ratio, 0],
                    [-1 / ratio, 0, 0],
                    [0, 0, 1]])
    return m2w @ (x_m - map_origin)


def world2map(map_origin, ratio, x_w):
    w2m = np.array([[0, -ratio, 0],
                    [ratio, 0, 0],
                    [0, 0, 1]])
    return w2m @ x_w + map_origin


def boundaries(robot_pose, fov_angle, distance, grid):
    global global_map
    global_map = grid
    x0, y0, theta = robot_pose
    x1 = x0 + distance * np.cos(theta - 0.5 * fov_angle)
    y1 = y0 + distance * np.sin(theta - 0.5 * fov_angle)
    x2 = x0 + distance * np.cos(theta + 0.5 * fov_angle)
    y2 = y0 + distance * np.sin(theta + 0.5 * fov_angle)

    # Set up the grid boundaries
    x1 = min(max(x1, 0), grid.shape[0] - 1)
    x2 = min(max(x2, 0), grid.shape[0] - 1)

    y1 = min(max(y1, 0), grid.shape[1] - 1)
    y2 = min(max(y2, 0), grid.shape[1] - 1)

    return [x1, y1], [x2, y2]


def SDF_RT(robot_pose, fov, radius, RT_res, grid, inner_r=0):
    global global_map
    global_map = grid
    pts = raytracing(robot_pose, fov, radius, RT_res, grid)
    in_1, in_2 = boundaries(robot_pose, fov, inner_r, grid)
    if inner_r == 0:
        pts = [in_1] + pts + [in_2]
    else:
        pts = [in_1] + pts + [in_2, in_1]
    return vertices_filter(np.array(pts))


def raytracing(robot_pose, fov, radius, RT_res, grid):
    x0, y0, theta = robot_pose
    out_1, out_2 = boundaries(robot_pose, fov, radius, grid)
    # y_mid = [y0 + radius * np.sin(theta - 0.5 * fov + i*fov / RT_res) for i in range(RT_res+1)]
    # x_mid = [x0 + radius * np.cos(theta - 0.5 * fov + i*fov / RT_res) for i in range(RT_res+1)]
    y_mid = np.linspace(out_1[1], out_2[1], RT_res)
    x_mid = np.linspace(out_1[0], out_2[0], RT_res)
    pts = []
    for i in range(len(x_mid)):
        xx, yy = DDA(int(x0), int(y0), int(x_mid[i]), int(y_mid[i]))
        if not pts or (yy != pts[-1][1] or xx != pts[-1][0]):
            pts.append([xx, yy])
    return pts


# @lru_cache(None)
def DDA(x0, y0, x1, y1):
    # find absolute differences
    dx = x1 - x0
    dy = y1 - y0

    # find maximum difference
    steps = int(max(abs(dx), abs(dy)))

    # calculate the increment in x and y
    x_inc = dx / steps
    y_inc = dy / steps

    # start with 1st point
    x = float(x0)
    y = float(y0)

    for i in range(steps):
        if 0 < int(x) < len(global_map) and 0 < int(y) < len(global_map[0]):
            if global_map[int(x), int(y)] == 1:
                break
        x = x + x_inc
        y = y + y_inc
    return int(x) + 1, int(y) + 1


def vertices_filter(polygon, angle_threshold=0.05):
    diff = polygon[1:] - polygon[:-1]
    diff_norm = np.sqrt(np.einsum('ij,ji->i', diff, diff.T))
    unit_vector = np.divide(diff, diff_norm[:, None], out=np.zeros_like(diff), where=diff_norm[:, None] != 0)
    angle_distance = np.round(np.einsum('ij,ji->i', unit_vector[:-1, :], unit_vector[1:, :].T), 5)
    angle_abs = np.abs(np.arccos(angle_distance))
    minimum_polygon = polygon[[True] + list(angle_abs > angle_threshold) + [True], :]
    return minimum_polygon


def polygon_SDF(polygon, point):
    N = len(polygon) - 1
    e = polygon[1:] - polygon[:-1]
    v = point - polygon[:-1]
    pq = v - e * np.clip((v[:, 0] * e[:, 0] + v[:, 1] * e[:, 1]) /
                         (e[:, 0] * e[:, 0] + e[:, 1] * e[:, 1]), 0, 1).reshape(N, -1)
    d = np.min(pq[:, 0] * pq[:, 0] + pq[:, 1] * pq[:, 1])
    wn = 0
    for i in range(N):
        val3 = np.cross(e[i], v[i])
        i2 = int(np.mod(i + 1, N))
        cond1 = 0 <= v[i, 1]
        cond2 = 0 > v[i2, 1]
        wn += 1 if cond1 and cond2 and val3 > 0 else 0
        wn -= 1 if ~cond1 and ~cond2 and val3 < 0 else 0
    sign = 1 if wn == 0 else -1
    return np.sqrt(d) * sign
