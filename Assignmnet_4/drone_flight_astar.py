import heapq
import matplotlib.pyplot as plt
import numpy as np

# Definisi grid terrain
terrain_map = [
    ['S', 1, 2, 3, '#', 5, 6],
    [1, '#', 2, 4, 5, '#', 7],
    [2, 2, 3, '#', 6, 7, 8],
    [3, '#', 4, 5, 6, '#', 9],
    [4, 5, 6, 7, 8, 9, 'G']
]

rows = len(terrain_map)
cols = len(terrain_map[0])

# Temukan posisi start dan goal
def find_position(symbol):
    for r in range(rows):
        for c in range(cols):
            if terrain_map[r][c] == symbol:
                return (r, c)
    return None

start = find_position('S')
goal = find_position('G')

# Dapatkan elevation cost
def elevation_cost(r, c):
    value = terrain_map[r][c]
    if value == '#':
        return float('inf')  # no-fly zone
    if value == 'S' or value == 'G':
        return 0
    return int(value)

# Heuristic: Manhattan Distance
def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

# Validasi gerak
def valid(r, c):
    return 0 <= r < rows and 0 <= c < cols and terrain_map[r][c] != '#'

# A* dengan elevation cost
def astar(start, goal):
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}
    visited = set()

    while open_set:
        _, cost, current = heapq.heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, current)

        visited.add(current)

        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor = (current[0] + dr, current[1] + dc)
            if valid(*neighbor):
                tentative_g_score = cost + elevation_cost(*neighbor)
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))

    return None

# Rekonstruksi jalur
def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

# Visualisasi
def visualize_path(path):
    temp_map = np.copy(np.array(terrain_map, dtype=object))
    for r, c in path:
        if temp_map[r][c] not in ['S', 'G']:
            temp_map[r][c] = '*'

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Drone Flight Path")

    for r in range(rows):
        for c in range(cols):
            value = temp_map[r][c]
            if value == 'S':
                color = 'orange'
            elif value == 'G':
                color = 'green'
            elif value == '#':
                color = 'black'
            elif value == '*':
                color = 'blue'
            else:
                color = plt.cm.terrain(int(value)/9)
            ax.add_patch(plt.Rectangle((c, rows-1-r), 1, 1, color=color, ec='gray'))

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.grid(True)
    plt.gca().invert_yaxis()
    plt.show()

# Testing
path = astar(start, goal)
if path:
    print("Flight path:", path)
    visualize_path(path)
else:
    print("No path found!")