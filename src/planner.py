import cv2
import numpy as np
import heapq

from semantics import CLASS_INDEX, DEFAULT_COSTS, UNKNOWN, FREE_COST

FREE_PROBABILITY = 0.2
OCCUPIED_PROBABILITY = 0.8
CLEARANCE_WEIGHT = 4.0
CLEARANCE_CELLS = 6
HEURISTIC_WEIGHT = 1.0
LETHAL_COST = 50.0

CELL_SIZE = 0.1
ROBOT_RADIUS = 0.45

SQRT2 = float(np.sqrt(2.0))


class Planner:
    def __init__(self, cell_size=CELL_SIZE, robot_radius=ROBOT_RADIUS):
        self.semantic_costs = dict(DEFAULT_COSTS)
        self.semantic_names = dict(CLASS_INDEX)
        self.cost_version = 0
        self.cost_table = None
        self.cost_table_version = -1
        self.cell_size = cell_size
        self.robot_radius = robot_radius
        self.inflation_cells = int(np.ceil(robot_radius / cell_size))
        self.kernels = {}
        self.last_inflation = self.inflation_cells

    def update_semantic_cost(self, command):
        for key in ("object", "action", "amount"):
            if key not in command:
                raise ValueError(f"Command is missing '{key}': {command}")

        obj = command["object"]
        action = command["action"]

        try:
            amount = float(command["amount"])
        except (TypeError, ValueError):
            raise ValueError(f"Amount must be a number, got {command['amount']!r}")

        if obj not in self.semantic_names:
            raise ValueError(f"Unknown semantic object: {obj}")
        semantic_id = self.semantic_names[obj]
        if action == "increase":

            self.semantic_costs[semantic_id] += amount

        elif action == "decrease":

            self.semantic_costs[semantic_id] = max(
                0.0,
                self.semantic_costs[semantic_id] - amount
            )

        elif action == "set":

            self.semantic_costs[semantic_id] = amount

        else:
            raise ValueError(f"Unknown action: {action}")

        self.cost_version += 1

    def heuristic(self, a, b):
        straight = abs(a[0] - b[0])
        across = abs(a[1] - b[1])
        return (straight + across) + (SQRT2 - 2.0) * min(straight, across)

    def astar(self, cost_map, start, goal):
        height, width = cost_map.shape
        start_row, start_col = int(start[0]), int(start[1])
        goal_row, goal_col = int(goal[0]), int(goal[1])

        if not (0 <= start_row < height and 0 <= start_col < width):
            return None
        if not (0 <= goal_row < height and 0 <= goal_col < width):
            return None

        costs = np.ascontiguousarray(cost_map, dtype=np.float64).ravel()
        passable = np.isfinite(costs)

        start_index = start_row * width + start_col
        goal_index = goal_row * width + goal_col

        if not passable[goal_index]:
            return None

        passable[start_index] = True

        cell_count = costs.size
        best_cost = np.full(cell_count, np.inf)
        came_from = np.full(cell_count, -1, dtype=np.int64)
        closed = np.zeros(cell_count, dtype=bool)

        best_cost[start_index] = 0.0
        open_set = [(0.0, start_index)]
        push = heapq.heappush
        pop = heapq.heappop

        while open_set:
            _, current = pop(open_set)

            if closed[current]:
                continue
            if current == goal_index:
                break

            closed[current] = True
            row, col = divmod(current, width)
            current_cost = best_cost[current]

            up = row > 0
            down = row < height - 1
            left = col > 0
            right = col < width - 1

            north = current - width
            south = current + width
            west = current - 1
            east = current + 1

            free_north = up and passable[north]
            free_south = down and passable[south]
            free_west = left and passable[west]
            free_east = right and passable[east]

            steps = (
                (north, 1.0, free_north),
                (south, 1.0, free_south),
                (west, 1.0, free_west),
                (east, 1.0, free_east),
                (north - 1, SQRT2, up and left and free_north and free_west),
                (north + 1, SQRT2, up and right and free_north and free_east),
                (south - 1, SQRT2, down and left and free_south and free_west),
                (south + 1, SQRT2, down and right and free_south and free_east),
            )

            for neighbor, step, reachable in steps:
                if not reachable or closed[neighbor] or not passable[neighbor]:
                    continue

                new_cost = current_cost + costs[neighbor] * step

                if new_cost < best_cost[neighbor]:
                    best_cost[neighbor] = new_cost
                    came_from[neighbor] = current

                    neighbor_row, neighbor_col = divmod(neighbor, width)
                    straight = abs(neighbor_row - goal_row)
                    across = abs(neighbor_col - goal_col)

                    push(open_set, (
                        new_cost + HEURISTIC_WEIGHT * (
                            (straight + across)
                            + (SQRT2 - 2.0) * min(straight, across)
                        ),
                        neighbor,
                    ))

        if goal_index != start_index and came_from[goal_index] < 0:
            return None

        path = []
        current = goal_index

        while current >= 0:
            row, col = divmod(int(current), width)
            path.append((row, col))
            current = came_from[current]

        path.reverse()
        return path

    def walk(self, start, goal):
        row, col = start
        goal_row, goal_col = goal

        straight = abs(goal_row - row)
        across = abs(goal_col - col)
        step_row = 1 if goal_row > row else -1
        step_col = 1 if goal_col > col else -1
        error = straight - across

        cells = [(row, col)]

        while row != goal_row or col != goal_col:
            doubled = 2 * error

            if doubled > -across:
                error -= across
                row += step_row
            if doubled < straight:
                error += straight
                col += step_col

            cells.append((row, col))

        return cells

    def visible(self, passable, start, goal):
        previous = start

        for cell in self.walk(start, goal)[1:]:
            if not passable[cell]:
                return False

            if cell[0] != previous[0] and cell[1] != previous[1]:
                if not passable[previous[0], cell[1]]:
                    return False
                if not passable[cell[0], previous[1]]:
                    return False

            previous = cell

        return True

    def smooth(self, path, passable):
        if len(path) < 3:
            return path

        corners = [path[0]]
        anchor = 0

        for index in range(2, len(path)):
            if not self.visible(passable, path[anchor], path[index]):
                anchor = index - 1
                corners.append(path[anchor])

        corners.append(path[-1])

        cells = [corners[0]]
        for index in range(1, len(corners)):
            cells.extend(self.walk(corners[index - 1], corners[index])[1:])

        return cells

    def find_neighbors(self, cell, cost_map):
        row, col = cell
        height, width = cost_map.shape
        neighbors = []

        def open_cell(r, c):
            return 0 <= r < height and 0 <= c < width and np.isfinite(cost_map[r, c])

        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if open_cell(row + dr, col + dc):
                neighbors.append((row + dr, col + dc))

        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            if not open_cell(row + dr, col + dc):
                continue
            if open_cell(row + dr, col) and open_cell(row, col + dc):
                neighbors.append((row + dr, col + dc))

        return neighbors

    def class_cost_table(self):
        if self.cost_table_version == self.cost_version:
            return self.cost_table

        class_ids = [key for key in self.semantic_costs if key >= 0]
        size = (max(class_ids) + 1) if class_ids else 0

        table = np.full(size, self.semantic_costs[UNKNOWN], dtype=np.float32)
        for class_id in class_ids:
            table[class_id] = self.semantic_costs[class_id]

        self.cost_table = table
        self.cost_table_version = self.cost_version
        return table

    def inflation_kernel(self, cells):
        if cells not in self.kernels:
            size = 2 * cells + 1
            self.kernels[cells] = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (size, size)
            )
        return self.kernels[cells]

    def lethal_mask(self, semantic_map, probability_grid, cells=None):
        if cells is None:
            cells = self.inflation_cells

        labelled = semantic_map >= 0
        table = self.class_cost_table()

        blocked = ~labelled & (probability_grid > OCCUPIED_PROBABILITY)

        if table.size and labelled.any():
            blocked[labelled] = table[semantic_map[labelled]] >= LETHAL_COST

        if cells <= 0 or not blocked.any():
            return blocked

        return cv2.dilate(
            blocked.astype(np.uint8), self.inflation_kernel(cells)
        ) > 0

    def inflation_schedule(self):
        schedule = [self.inflation_cells]
        if self.inflation_cells > 1:
            schedule.append(1)
        if self.inflation_cells > 0:
            schedule.append(0)
        return schedule

    def semantic_to_cost_map(self, semantic_map, probability_grid, cells=None):
        cost_map = np.full(
            semantic_map.shape,
            self.semantic_costs[UNKNOWN],
            dtype=np.float32
        )

        cost_map[probability_grid < FREE_PROBABILITY] = FREE_COST

        table = self.class_cost_table()

        if table.size:
            labelled = semantic_map >= 0
            if labelled.any():
                cost_map[labelled] = table[semantic_map[labelled]]

        lethal = self.lethal_mask(semantic_map, probability_grid, cells)
        cost_map = self.distance_gradient_cost(cost_map, lethal)

        cost_map[lethal] = np.inf

        return cost_map

    def plan(self, semantic_map, probability_grid, start, goal):
        for cells in self.inflation_schedule():
            cost_map = self.semantic_to_cost_map(
                semantic_map, probability_grid, cells
            )
            path = self.astar(cost_map, start, goal)

            if path:
                self.last_inflation = cells
                return self.smooth(path, np.isfinite(cost_map))

        return None

    def distance_gradient_cost(self, cost_map, lethal):
        if not lethal.any():
            return cost_map

        distances = cv2.distanceTransform(
            (~lethal).astype(np.uint8), cv2.DIST_L2, 5
        )

        near = distances < CLEARANCE_CELLS
        falloff = 1.0 - distances[near] / CLEARANCE_CELLS
        cost_map[near] += CLEARANCE_WEIGHT * falloff * falloff

        return cost_map
