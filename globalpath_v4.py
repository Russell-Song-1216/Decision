import folium
from folium.plugins import Draw, LocateControl
import numpy as np
import json
import time
from pynmea2 import NMEASentence
from pynmea2 import TalkerSentence
from decimal import Decimal
import math
from scipy.interpolate import splprep, splev
from datetime import datetime
import heapq

# 定義曼哈頓距離啟發函數
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# 計算兩個經緯度之間的地球表面距離，使用 Haversine 公式
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # 地球半徑，單位為米
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance


# 修改 get_neighbors 函數，加入對障礙物的避讓
def get_neighbors(node, grid, obstacles, safety_radius_meters=3):
    neighbors = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for dx, dy in directions:
        neighbor = (node[0] + dx, node[1] + dy)

        # 檢查鄰點是否在網格內
        if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
            is_safe = True

            # 與障礙物保持安全距離
            for obstacle in obstacles:
                distance = np.linalg.norm(np.array(neighbor) - np.array(obstacle))
                if distance < safety_radius_meters:
                    is_safe = False
                    break

            if is_safe:
                neighbors.append(neighbor)

    return neighbors


def bidirectional_astar(grid, start, goal, obstacles):
    open_set_start = []
    heapq.heappush(open_set_start, (0, start))
    came_from_start = {}
    gscore_start = {start: 0}
    fscore_start = {start: heuristic(start, goal)}

    open_set_goal = []
    heapq.heappush(open_set_goal, (0, goal))
    came_from_goal = {}
    gscore_goal = {goal: 0}
    fscore_goal = {goal: heuristic(goal, start)}

    closed_set_start = set()
    closed_set_goal = set()

    while open_set_start and open_set_goal:
        _, current_start = heapq.heappop(open_set_start)
        closed_set_start.add(current_start)
        if current_start in closed_set_goal:
            return reconstruct_path(came_from_start, came_from_goal, current_start)

        for neighbor in get_neighbors(current_start, grid, obstacles):
            tentative_g_score = gscore_start[current_start] + heuristic(current_start, neighbor)
            if tentative_g_score < gscore_start.get(neighbor, float("inf")):
                came_from_start[neighbor] = current_start
                gscore_start[neighbor] = tentative_g_score
                fscore_start[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set_start, (fscore_start[neighbor], neighbor))

        _, current_goal = heapq.heappop(open_set_goal)
        closed_set_goal.add(current_goal)
        if current_goal in closed_set_start:
            return reconstruct_path(came_from_start, came_from_goal, current_goal)

        for neighbor in get_neighbors(current_goal, grid, obstacles):
            tentative_g_score = gscore_goal[current_goal] + heuristic(current_goal, neighbor)
            if tentative_g_score < gscore_goal.get(neighbor, float("inf")):
                came_from_goal[neighbor] = current_goal
                gscore_goal[neighbor] = tentative_g_score
                fscore_goal[neighbor] = tentative_g_score + heuristic(neighbor, start)
                heapq.heappush(open_set_goal, (fscore_goal[neighbor], neighbor))

    return None

def reconstruct_path(came_from_start, came_from_goal, meeting_point):
    path_start = []
    current = meeting_point
    while current in came_from_start:
        path_start.append(current)
        current = came_from_start[current]
    path_start = path_start[::-1]
    path_goal = []
    current = meeting_point
    while current in came_from_goal:
        current = came_from_goal[current]
        path_goal.append(current)
    return path_start + path_goal

# 經緯度轉換為網格座標
def latlng_to_grid(latlng, grid_size=(8000, 8000), map_bounds=((22.85, 120.19), (22.88, 120.22))):
    lat, lng = latlng
    lat_min, lng_min = map_bounds[0]
    lat_max, lng_max = map_bounds[1]
    x = int((lat - lat_min) / (lat_max - lat_min) * grid_size[0])
    y = int((lng - lng_min) / (lng_max - lng_min) * grid_size[1])
    return (x, y)

# 網格座標轉換為經緯度
def grid_to_latlng(grid_pos, grid_size=(8000, 8000), map_bounds=((22.85, 120.19), (22.88, 120.22))):
    x, y = grid_pos
    lat_min, lng_min = map_bounds[0]
    lat_max, lng_max = map_bounds[1]
    lat = x / grid_size[0] * (lat_max - lat_min) + lat_min
    lng = y / grid_size[1] * (lng_max - lng_min) + lng_min
    return (lat, lng)

# 計算路徑總長度，單位為米
def calculate_total_length(path_points):
    total_length = 0
    for i in range(len(path_points) - 1):
        lat1, lon1 = path_points[i]         # 拆分當前點的緯度和經度
        lat2, lon2 = path_points[i + 1]     # 拆分下一個點的緯度和經度
        total_length += haversine_distance(lat1, lon1, lat2, lon2)  # 計算兩點間的距離
    return total_length

# 卡塔穆羅姆樣條曲線平滑處理
def catmull_rom_spline(points, n_points=300):
    points = np.array(points)
    if len(points) < 4:
        print("點數不足，無法進行樣條擬合，返回原始路徑。")
        return points
    tck, u = splprep([points[:, 0], points[:, 1]], k=2, s=0.0000000001)
    u_new = np.linspace(u.min(), u.max(), n_points)
    x_new, y_new = splev(u_new, tck)
    return np.vstack((x_new, y_new)).T

# 計算兩點之間的方向向量，並基於之前的向量進行補正
def calculate_direction_vector(point1, point2):
    # 計算從 point1 到 point2 的方向向量
    direction_vector = np.array([point2[0] - point1[0], point2[1] - point1[1]])

    # 將向量進行單位化處理
    direction_vector = direction_vector / np.linalg.norm(direction_vector)

    return direction_vector

def generate_points_every_meter(start_point, end_point):

    distance = haversine_distance(start_point[0], start_point[1], end_point[0], end_point[1])
    num_points = int(distance/4.95)  # 每5米一個點
    if num_points < 2:
        return [start_point, end_point]

    lat1, lon1 = map(np.radians, start_point)
    lat2, lon2 = map(np.radians, end_point)

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    points = []
    for i in range(num_points + 1):
        fraction = i / num_points
        lat = lat1 + fraction * delta_lat
        lon = lon1 + fraction * delta_lon
        points.append((np.degrees(lat), np.degrees(lon)))

    return points

# 主程式
def calculate_route_json(start_point, goal_point_1, obstacles=None):
    map_bounds = ((22.85, 120.19), (22.88, 120.22))
    grid_size = (4000, 4000)
    grid = np.zeros(grid_size)

    start_grid = latlng_to_grid(start_point, grid_size, map_bounds)
    goal_grid_1 = latlng_to_grid(goal_point_1, grid_size, map_bounds)


    if obstacles is not None:
        obstacles = [latlng_to_grid(obs, grid_size, map_bounds) for obs in obstacles]
    else:
        obstacles = []

    if not obstacles:
        # print("無障礙物，計算直線路徑，並生成每米座標點。")
        full_path = [start_grid, goal_grid_1]
        path_latlng = [grid_to_latlng((int(x), int(y)), grid_size, map_bounds) for x, y in full_path]
        # 使用每米生成座標點的邏輯
        if len(path_latlng) == 2:
            smooth_path = generate_points_every_meter(path_latlng[0], path_latlng[1])
        else:
            smooth_path = path_latlng
    else:
        full_path = bidirectional_astar(grid, start_grid, goal_grid_1, obstacles)
        if not full_path:
            print("沒有找到可行路徑。")
            return None
        path_latlng = [grid_to_latlng((int(x), int(y)), grid_size, map_bounds) for x, y in full_path]
        total_length = calculate_total_length(path_latlng)
        n_points = int(total_length)
        if len(path_latlng) < 4:
            smooth_path = path_latlng
        else:
            smooth_path = catmull_rom_spline(np.array(path_latlng), n_points=n_points)

    # 保存平滑路徑為 JSON 文件
    # with open("generated_path.json", "w") as f:
    #     json.dump(smooth_path, f, indent=4)
    # print("平滑後的路徑已保存為 generated_path.json")
    # print(smooth_path[1][0])
    target=[None,None]
    target[0],target[1] = smooth_path[2][0],smooth_path[2][1]
    return target

# calculate_route_json((22.865271, 120.204960),(22.864025, 120.201827))

