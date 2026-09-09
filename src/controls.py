import numpy as np


class PathFollower:
    def __init__(self, lookahead=0.8, max_linear=1.0, max_angular=2.0,
                 goal_tolerance=0.3, curvature_gain=1.5):
        self.lookahead = lookahead
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.goal_tolerance = goal_tolerance
        self.curvature_gain = curvature_gain
        self.index = 0

    def reset(self):
        self.index = 0

    def snap(self, position, path):
        if len(path) == 0:
            self.index = 0
            return

        offsets = path - position
        self.index = int(np.argmin(np.einsum("ij,ij->i", offsets, offsets)))

    def reached(self, position, path):
        if len(path) == 0:
            return True
        return np.linalg.norm(path[-1] - position) < self.goal_tolerance

    def lookahead_point(self, position, path):
        if self.index >= len(path):
            self.index = len(path) - 1

        while self.index < len(path) - 1:
            if np.linalg.norm(path[self.index] - position) < self.lookahead:
                self.index += 1
            else:
                break

        return path[self.index]

    def command(self, pose, path):
        if len(path) == 0:
            return 0.0, 0.0

        x, y, yaw = pose
        position = np.array([x, y])

        if self.reached(position, path):
            return 0.0, 0.0

        offset = self.lookahead_point(position, path) - position
        distance = float(np.linalg.norm(offset))

        if distance < 1e-6:
            return 0.0, 0.0

        forward = np.cos(yaw) * offset[0] + np.sin(yaw) * offset[1]
        lateral = -np.sin(yaw) * offset[0] + np.cos(yaw) * offset[1]

        if forward <= 0.0:
            turn = self.max_angular if lateral >= 0.0 else -self.max_angular
            return 0.0, turn

        curvature = 2.0 * lateral / (distance * distance)
        linear = self.max_linear / (1.0 + self.curvature_gain * abs(curvature))
        angular = float(np.clip(
            linear * curvature, -self.max_angular, self.max_angular
        ))

        return linear, angular
