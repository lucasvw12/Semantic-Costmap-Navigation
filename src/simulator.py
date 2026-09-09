import mujoco
import mujoco.viewer
import threading
import time
from pathlib import Path
import geometry
from renderer import Renderer, WIDTH, HEIGHT
import cv2
import numpy as np
from planner import Planner
from perception import Perception
from geometry import pixel_to_3d, camera_to_world, ground_points
from mapping import Mapping, OccupancyGrid
from llm_cost_parser import LLMParser
from robot import Robot
from controls import PathFollower
from semantics import CLASSES, PROMPT, label_to_class_id

INSTRUCTIONS = 'N/A'

CELL = 0.1
BOUND = 25.0

GOAL = (12.0, -4.0)
MAX_LINEAR = 0.5
MAX_ANGULAR = 2.0
LOOKAHEAD = 0.4
GOAL_TOLERANCE = 0.3

LABEL_DECAY = 0.98
LABEL_MIN_SCORE = 0.25
INSTRUCTION_KEY = ord('i')

OBSTACLE_MIN_Z = 0.05
OBSTACLE_MAX_Z = 1.20
OBSTACLE_MAX_RANGE = 10.0

GROUND_MAX_RANGE = 8.0
GROUND_STRIDE = 4
GROUND_Z = 0.10
GROUND_CONFIDENCE = 0.8

VIEW_CAMERA = "top_down"
SHOW_PATH = True
PATH_STRIDE = 8
PATH_HEIGHT = 0.12
PATH_WIDTH = 0.05
PATH_RGBA = (0.10, 0.82, 0.72, 1.0)
GOAL_RADIUS = 0.35
GOAL_RGBA = (1.00, 0.45, 0.10, 1.0)

SURVEY_ROTATIONS = 1.0
SURVEY_ANGULAR = 0.6
SURVEY_MIN_OBSERVATIONS = 2
SURVEY_TIMEOUT = 40.0

CONTROL_RATE = 120.0
MAX_CONTROL_DT = 0.05
SENSE_INTERVAL = 0.1
DISPLAY_INTERVAL = 1.0 / 15.0
PLAN_INTERVAL = 3.0


class PerceptionWorker:
    def __init__(self, extract):
        self.extract = extract
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.job = None
        self.results = []
        self.ready = False
        self.running = True
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def submit(self, job):
        with self.lock:
            if not self.ready or self.job is not None:
                return False
            self.job = job

        self.wake.set()
        return True

    def drain(self):
        with self.lock:
            results = self.results
            self.results = []
        return results

    def is_ready(self):
        with self.lock:
            return self.ready

    def stop(self):
        self.running = False
        self.wake.set()

    def run(self):
        started = time.perf_counter()
        perception = Perception()
        perception.warmup(PROMPT, WIDTH, HEIGHT)
        print("perception ready on %s in %.1fs"
              % (perception.device, time.perf_counter() - started))

        with self.lock:
            self.ready = True

        while self.running:
            if not self.wake.wait(0.2):
                continue

            self.wake.clear()

            with self.lock:
                job = self.job

            if job is None:
                continue

            elapsed = time.perf_counter()

            try:
                observations = self.extract(perception, *job)
            except Exception as error:
                print("perception failed: %s" % error)
                observations = []

            elapsed = time.perf_counter() - elapsed

            with self.lock:
                self.job = None
                self.results.append((observations, elapsed))


class PlannerWorker:
    def __init__(self, planner):
        self.planner = planner
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.job = None
        self.commands = []
        self.results = []
        self.running = True
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def submit(self, job):
        with self.lock:
            if self.job is not None:
                return False
            self.job = job

        self.wake.set()
        return True

    def apply_command(self, command):
        with self.lock:
            self.commands.append(command)

        self.wake.set()

    def drain(self):
        with self.lock:
            results = self.results
            self.results = []
        return results

    def stop(self):
        self.running = False
        self.wake.set()

    def run(self):
        while self.running:
            if not self.wake.wait(0.2):
                continue

            self.wake.clear()

            with self.lock:
                commands = self.commands
                self.commands = []
                job = self.job

            for command in commands:
                try:
                    self.planner.update_semantic_cost(command)
                except Exception as error:
                    print("instruction rejected: %s" % error)

            if job is None:
                continue

            started = time.perf_counter()

            try:
                cells = self.planner.plan(*job)
            except Exception as error:
                print("planning failed: %s" % error)
                cells = None

            elapsed = time.perf_counter() - started

            with self.lock:
                self.job = None
                self.results.append((
                    cells,
                    job[2],
                    self.planner.last_inflation,
                    self.planner.cost_version,
                    elapsed
                ))


class InstructionReader:
    def __init__(self):
        self.lock = threading.Lock()
        self.commands = []
        self.busy = False
        self.parser = None

    def drain(self):
        with self.lock:
            commands = self.commands
            self.commands = []
        return commands

    def start(self, text=None):
        with self.lock:
            if self.busy:
                return
            self.busy = True

        threading.Thread(target=self.run, args=(text,), daemon=True).start()

    def run(self, text):
        try:
            if text is None:
                try:
                    text = input("instruction: ").strip()
                except EOFError:
                    text = ""

            if text:
                if self.parser is None:
                    print("loading instruction parser...")
                    self.parser = LLMParser()

                command = self.parser.parse_instruction(text)

                with self.lock:
                    self.commands.append(command)
        except Exception as error:
            print("instruction rejected: %s" % error)
        finally:
            with self.lock:
                self.busy = False


class Simulator:
    def __init__(self):
        scene_path = Path(__file__).parent.parent / "assets" / "scene.xml"
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.camera_id = self.model.camera("rgbd_camera").id

        try:
            self.view_camera_id = self.model.camera(VIEW_CAMERA).id
        except KeyError:
            print("view camera %r not in the scene, showing the robot camera"
                  % VIEW_CAMERA)
            self.view_camera_id = self.camera_id

        self.K = geometry.intrinsic(
            WIDTH,
            HEIGHT,
            self.model.cam_fovy[self.camera_id]
        )

    def draw_path(self, viewer, path_world, goal):
        scene = viewer.user_scn
        scene.ngeom = 0

        if not SHOW_PATH:
            return

        identity = np.eye(3).ravel()
        blank = np.zeros(3)

        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([GOAL_RADIUS, 0.0, 0.0]),
            np.array([goal[0], goal[1], PATH_HEIGHT]),
            identity,
            np.array(GOAL_RGBA, dtype=np.float32)
        )
        scene.ngeom += 1

        if path_world is None or len(path_world) < 2:
            return

        points = path_world[::PATH_STRIDE]

        if not np.array_equal(points[-1], path_world[-1]):
            points = np.vstack([points, path_world[-1]])

        room = min(len(points), scene.maxgeom - scene.ngeom + 1)

        for index in range(1, room):
            geom = scene.geoms[scene.ngeom]

            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                blank,
                blank,
                identity,
                np.array(PATH_RGBA, dtype=np.float32)
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                PATH_WIDTH,
                np.array([points[index - 1][0], points[index - 1][1], PATH_HEIGHT]),
                np.array([points[index][0], points[index][1], PATH_HEIGHT])
            )

            scene.ngeom += 1

    def cell_of(self, mapping, x, y):
        rows, cols = mapping.world_to_grid(x=np.array([x]), y=np.array([y]))
        return int(rows[0]), int(cols[0])

    def mark_ground_free(self, grid, mapping, depth_image,
                         camera_rotation, camera_coordinates):
        floor_points = ground_points(
            depth_image,
            self.K,
            camera_rotation,
            camera_coordinates,
            stride=GROUND_STRIDE,
            max_range=GROUND_MAX_RANGE,
            ground_z=GROUND_Z
        )

        if len(floor_points) == 0:
            return 0

        rows, cols = mapping.world_to_grid(
            x=floor_points[:, 0],
            y=floor_points[:, 1]
        )

        grid.update_free_cells(
            rows,
            cols,
            np.full(len(rows), GROUND_CONFIDENCE, dtype=np.float32),
            protect=grid.labelled_mask()
        )

        return len(floor_points)

    def collect_obstacles(self, perception, rgb_image, depth_image,
                          camera_rotation, camera_coordinates):
        detected_objects = perception.detect(rgb_image, PROMPT)
        boxes = detected_objects[0]["boxes"]

        if len(boxes) == 0:
            return []

        segmentation_coordinates = perception.segment(rgb_image, boxes)
        text_labels = detected_objects[0]["text_labels"]
        scores = detected_objects[0]["scores"]

        observations = []

        for object_index, object_coordinates in enumerate(
            segmentation_coordinates
        ):
            x = object_coordinates[:, 0].astype(int)
            y = object_coordinates[:, 1].astype(int)

            valid_pixels = (
                (x >= 0) &
                (x < depth_image.shape[1]) &
                (y >= 0) &
                (y < depth_image.shape[0])
            )

            x = x[valid_pixels]
            y = y[valid_pixels]

            if len(x) == 0:
                continue

            depths = depth_image[y, x]
            usable = (depths > 0.1) & (depths < GROUND_MAX_RANGE * 2)

            x = x[usable]
            y = y[usable]
            depths = depths[usable]

            if len(x) == 0:
                continue

            points_camera = pixel_to_3d(x, y, depths, self.K)

            points_world = camera_to_world(
                points_camera,
                camera_rotation,
                camera_coordinates
            )

            offsets = points_world[:, :2] - camera_coordinates[:2]
            ranges = np.linalg.norm(offsets, axis=1)

            keep = (
                (points_world[:, 2] > OBSTACLE_MIN_Z) &
                (points_world[:, 2] < OBSTACLE_MAX_Z) &
                (ranges < OBSTACLE_MAX_RANGE)
            )

            points_world = points_world[keep]

            if len(points_world) == 0:
                continue

            observations.append((
                points_world,
                scores[object_index].item(),
                label_to_class_id(text_labels[object_index])
            ))

        return observations

    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            renderer = Renderer(self.model, self.data)
            planner = Planner(cell_size=CELL)
            plan_worker = PlannerWorker(planner)
            instructions = InstructionReader()
            worker = PerceptionWorker(self.collect_obstacles)

            if INSTRUCTIONS and INSTRUCTIONS != 'N/A':
                instructions.start(INSTRUCTIONS)

            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = self.view_camera_id

            robot = Robot(self.model, self.data)
            follower = PathFollower(
                lookahead=LOOKAHEAD,
                max_linear=MAX_LINEAR,
                max_angular=MAX_ANGULAR,
                goal_tolerance=GOAL_TOLERANCE
            )

            grid = OccupancyGrid(
                height=int(round(2 * BOUND / CELL)),
                width=int(round(2 * BOUND / CELL)),
                n_classes=len(CLASSES),
                label_decay=LABEL_DECAY,
                label_min_score=LABEL_MIN_SCORE
            )
            mapping = Mapping(CELL, x_min=-BOUND, y_min=-BOUND,
                              height=grid.height, width=grid.width)
            goal_cell = self.cell_of(mapping, GOAL[0], GOAL[1])

            path_world = None
            arrived = False
            last_planned_version = -1
            replan_requested = True
            free_count = 0
            reported_inflation = planner.inflation_cells
            surveying = True
            survey_remaining = 2.0 * np.pi * SURVEY_ROTATIONS
            survey_observations = 0
            rgb_image = None
            depth_image = None

            now = time.perf_counter()
            last_control = now
            next_sense = now
            next_display = now
            next_plan = now
            survey_started = None

            control_period = 1.0 / CONTROL_RATE

            while viewer.is_running():
                now = time.perf_counter()
                control_dt = min(now - last_control, MAX_CONTROL_DT)
                last_control = now

                if surveying:
                    robot.apply(0.0, SURVEY_ANGULAR, control_dt)
                    survey_remaining -= SURVEY_ANGULAR * control_dt

                    if survey_started is None and worker.is_ready():
                        survey_started = now

                    turned = survey_remaining <= 0.0
                    looked = survey_observations >= SURVEY_MIN_OBSERVATIONS
                    expired = (survey_started is not None
                               and now - survey_started > SURVEY_TIMEOUT)

                    if (turned and looked) or expired:
                        surveying = False
                        print("survey done after %.0fs: %d observations, "
                              "%d cells labelled%s" % (
                                  now - (survey_started or now),
                                  survey_observations,
                                  int(grid.labelled_mask().sum()),
                                  " (timed out)" if expired and not looked else ""
                              ))

                elif path_world is not None and not arrived:
                    linear, angular = follower.command(robot.pose, path_world)
                    robot.apply(linear, angular, control_dt)

                    if follower.reached(robot.position, path_world):
                        arrived = True
                        print("goal reached at (%.2f, %.2f)" % (robot.x, robot.y))

                mujoco.mj_forward(self.model, self.data)

                sensing = now >= next_sense
                displaying = now >= next_display

                if sensing or displaying:
                    rgb_image, depth_image, _ = renderer.render_frames(
                        want_depth=sensing
                    )

                if sensing:
                    next_sense = now + SENSE_INTERVAL

                    camera_coordinates = self.data.cam_xpos[self.camera_id].copy()
                    camera_rotation = self.data.cam_xmat[
                        self.camera_id
                    ].reshape(3, 3).copy()

                    free_count = self.mark_ground_free(
                        grid,
                        mapping,
                        depth_image,
                        camera_rotation,
                        camera_coordinates
                    )

                    worker.submit((
                        rgb_image,
                        depth_image,
                        camera_rotation,
                        camera_coordinates
                    ))

                for observations, elapsed in worker.drain():
                    survey_observations += 1
                    grid.decay_class_scores()

                    for points_world, confidence, class_id in observations:
                        rows, cols = mapping.world_to_grid(
                            x=points_world[:, 0],
                            y=points_world[:, 1]
                        )

                        grid.update_cells(
                            rows,
                            cols,
                            np.full(len(rows), confidence, dtype=np.float32),
                            class_id
                        )

                    replan_requested = True

                    print("x=%.2f y=%.2f yaw=%+.2f  free=%d  %.1fs  objects=%s" % (
                        robot.x,
                        robot.y,
                        robot.yaw,
                        free_count,
                        elapsed,
                        [
                            (CLASSES[c] if c >= 0 else "unknown", len(p))
                            for p, _, c in observations
                        ]
                    ))

                for command in instructions.drain():
                    plan_worker.apply_command(command)
                    replan_requested = True

                if displaying and rgb_image is not None:
                    next_display = now + DISPLAY_INTERVAL

                    cv2.imshow(
                        "RGB Image",
                        cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                    )

                    if (cv2.waitKey(1) & 0xFF) == INSTRUCTION_KEY:
                        instructions.start()

                if not arrived and (replan_requested or now >= next_plan):
                    start_cell = self.cell_of(mapping, robot.x, robot.y)

                    if mapping.contains(*start_cell) and plan_worker.submit((
                        grid.get_label_grid(),
                        grid.get_probability_grid(),
                        start_cell,
                        goal_cell
                    )):
                        next_plan = now + PLAN_INTERVAL
                        replan_requested = False

                for result in plan_worker.drain():
                    cells, start_cell, inflation, version, elapsed = result
                    if cells and inflation != reported_inflation:
                        reported_inflation = inflation
                        print("clearance margin now %.2f m%s" % (
                            reported_inflation * CELL,
                            "" if reported_inflation >= planner.inflation_cells
                            else "  (squeezing past an obstacle)"
                        ))

                    if cells:
                        rows = np.array([cell[0] for cell in cells])
                        cols = np.array([cell[1] for cell in cells])
                        xs, ys = mapping.grid_to_world(row=rows, col=cols)
                        path_world = np.column_stack([xs, ys])
                        follower.snap(robot.position, path_world)
                    elif path_world is not None:
                        path_world = None
                        print("no path from %s to %s - holding position"
                              % (start_cell, goal_cell))

                    if version != last_planned_version:
                        last_planned_version = version
                        print("replan v%d  %s -> %s  %s  %.0fms" % (
                            version,
                            start_cell,
                            goal_cell,
                            "%d cells" % len(cells) if cells else "no path",
                            elapsed * 1000.0
                        ))

                    self.draw_path(viewer, path_world, GOAL)

                viewer.sync()

                remaining = control_period - (time.perf_counter() - now)
                if remaining > 0:
                    time.sleep(remaining)

            worker.stop()
            plan_worker.stop()
            cv2.destroyAllWindows()
