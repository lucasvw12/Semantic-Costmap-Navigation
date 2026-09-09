import numpy as np


class Robot:
    def __init__(self, model, data, body_name="robot"):
        self.model = model
        self.data = data
        self.body_id = model.body(body_name).id
        self.qpos_address = model.jnt_qposadr[model.body(body_name).jntadr[0]]

        start = data.qpos[self.qpos_address:self.qpos_address + 3]
        self.x = float(start[0])
        self.y = float(start[1])
        self.z = float(start[2])
        self.yaw = 0.0

        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

        self.write()

    @property
    def pose(self):
        return self.x, self.y, self.yaw

    @property
    def position(self):
        return np.array([self.x, self.y])

    def apply(self, linear, angular, dt):
        self.linear_velocity = linear
        self.angular_velocity = angular

        self.yaw += angular * dt
        self.yaw = (self.yaw + np.pi) % (2 * np.pi) - np.pi

        self.x += linear * np.cos(self.yaw) * dt
        self.y += linear * np.sin(self.yaw) * dt

        self.write()

    def write(self):
        address = self.qpos_address
        self.data.qpos[address:address + 3] = (self.x, self.y, self.z)
        self.data.qpos[address + 3:address + 7] = (
            np.cos(self.yaw / 2.0), 0.0, 0.0, np.sin(self.yaw / 2.0)
        )
