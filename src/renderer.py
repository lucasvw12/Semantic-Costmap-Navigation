import mujoco
import numpy as np

WIDTH = 640
HEIGHT = 480


class Renderer:
    def __init__(self, model, data, width=WIDTH, height=HEIGHT):
        self.model = model
        self.data = data
        self.renderer = mujoco.Renderer(model, height=height, width=width)

    def render_frames(self, want_depth=True, want_segmentation=False):
        self.renderer.disable_segmentation_rendering()
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera="rgbd_camera")
        rgb = self.renderer.render()

        depth = None
        if want_depth:
            self.renderer.enable_depth_rendering()
            depth = self.renderer.render()
            self.renderer.disable_depth_rendering()

        segmentation = None
        if want_segmentation:
            self.renderer.enable_segmentation_rendering()
            segmentation = self.renderer.render()
            self.renderer.disable_segmentation_rendering()

        return rgb, depth, segmentation
