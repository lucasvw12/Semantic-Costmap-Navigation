# Semantic-Costmap-Navigation
# Semantic Costmap Navigation

> A simulated mobile robot that drives across a campus scene by naming what it sees, turning an open-vocabulary detector's output into a semantic costmap that A* plans across, and whose traversal costs you can change by talking to it. For example, "avoid barrels" raises the cost of barrels and makes the robot find the best path that keeps away from them.

---

## Table of Contents
- [About the Project](#about-the-project)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Known Issues and Roadmap](#known-issues-and-roadmap)
- [License](#license)
- [Contact](#contact)

---

## About the Project

Most occupancy grids answer one question: is this cell blocked? This project instead asks what is this cell? and assigns each class a configurable cost.

A differential-drive robot with an RGB-D camera navigates a MuJoCo scene containing crates, barrels, benches, trees, fences, and cars. Depth is deprojected into world points, while GroundingDINO and SAM detect and segment objects. These labels are fused into a 500 × 500 log-odds grid at 0.1 m resolution (cell width). A costmap converts labels to costs, inflates obstacles by the robot radius, and A* plans an 8-connected path. The robot follows the path using Pure Pursuit to reach the destination.

A local LLM can modify costs from natural-language commands such as "avoid barrels" or "make cars cost 50," changing the route without retraining.

### Key Features

- **Open-vocabulary perception**: the class list is a prompt string, not a trained head. Adding "fire hydrant" to `semantics.py` needs no retraining.
- **Semantic log-odds mapping**: obstacle and free-space evidence accumulate in Bayesian log-odds, with a separate per-class score layer that decays over time so stale labels fade.
- **Costmap planning with real clearance**: obstacles are inflated by the robot radius and made genuinely impassable, with a fallback schedule that relaxes the margin rather than giving up, plus line-of-sight path smoothing.
- **Talk to the costmap**: a locally run Qwen2.5-1.5B parses natural-language instructions into an object, action and amount, then replans on the spot.
- **Everything slow runs off the main loop**: perception and planning each own a thread, so control holds 120 Hz and rendering 15 Hz while a 6-second detection is in flight.
- **A survey before it commits**: the robot turns a full circle at startup and waits for real observations before choosing a route, which is worth roughly nine times more map than planning from an empty grid.

---

## Tech Stack

- **Simulation**: MuJoCo 3.10 for physics, offscreen RGB-D rendering and the passive viewer
- **Perception**: PyTorch 2.13 (CPU) and Transformers 5.14, running GroundingDINO-tiny for detection and SlimSAM-50 for segmentation
- **Language**: Qwen2.5-1.5B-Instruct, run locally for instruction parsing
- **Mapping and planning**: NumPy 2.5, OpenCV 5.0 for the distance transform and morphology, and a hand-written 8-connected A*
- **Runtime**: Python 3.12, using threads for the perception and planner workers

---

## Getting Started

Follow these steps to set up the project locally for development and testing.

### Prerequisites

- Python 3.12 or higher
- A GPU is optional. Everything runs on CPU, but a CUDA build of PyTorch makes perception several times faster.
- About 5 GB of free disk for the Hugging Face model cache, since the three models download on first run
- A working OpenGL context for the MuJoCo viewer

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/lucasvw12/Semantic-Costmap-Navigation.git
   cd Semantic-Costmap-Navigation
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   On macOS or Linux, use `source .venv/bin/activate` instead. Then:

   ```bash
   pip install mujoco torch transformers opencv-python numpy pillow accelerate
   ```

   Note that `requirements.txt` is currently empty. Once your environment works, run `pip freeze > requirements.txt` so the command above becomes `pip install -r requirements.txt`.

3. Environment variables:

   None are required. You can optionally set a Hugging Face token to avoid rate-limited downloads:

   ```env
   HF_TOKEN=your_huggingface_token
   ```

4. Run it:

   ```bash
   python src/main.py
   ```

   The three models download on first launch, so expect a few minutes before anything moves.

---

## Usage

Running `src/main.py` opens two windows: the MuJoCo viewer, pinned to a top-down camera with the planned route drawn into the scene, and an OpenCV window showing the robot's own RGB feed.

The robot surveys its surroundings, then drives from its spawn in the courtyard's back-left corner to the goal, replanning as the map fills in. Console output reports each perception cycle and each replan.

Press `i` in the RGB window to type an instruction. Parsing runs on a background thread, so the robot keeps driving while it thinks:

```
instruction: avoid barrels
replan v1  (337, 151) to (210, 370)  228 cells  412ms

instruction: make cars cost 50
```

To set an instruction before launch instead, edit `INSTRUCTIONS` at the top of `src/simulator.py`.

### Key Parameters

All tuning lives in module-level constants, with no config files.

| Constant | File | Default | Description |
| :--- | :--- | :--- | :--- |
| `GOAL` | `simulator.py` | `(12.0, -4.0)` | Target position in world metres |
| `MAX_LINEAR` | `simulator.py` | `0.5` | Drive speed in metres per second |
| `SENSE_INTERVAL` | `simulator.py` | `0.1` | Depth sensing period in seconds |
| `PLAN_INTERVAL` | `simulator.py` | `3.0` | Minimum seconds between replans |
| `OBSTACLE_MAX_Z` | `simulator.py` | `1.20` | Top of the height band treated as obstacle |
| `SURVEY_MIN_OBSERVATIONS` | `simulator.py` | `2` | Detections required before the robot moves |
| `CELL` | `simulator.py` | `0.1` | Grid cell width in metres |
| `ROBOT_RADIUS` | `planner.py` | `0.45` | Obstacle inflation radius in metres |
| `LETHAL_COST` | `planner.py` | `50.0` | Semantic cost at which a class becomes impassable |
| `CLASSES` | `semantics.py` | 11 classes | The detector prompt and the cost table's keys |
| `USE_SAM` | `perception.py` | `True` | Set to False to skip segmentation and halve the cycle time |

---

## Known Issues and Roadmap

- Obstacles clear their own footprint. `ground_points` accepts any return whose height is within 0.10 m of the floor, and the bottom 10 cm of every vertical surface satisfies that, so each obstacle marks its own base as free floor at 10 Hz. The fix is to reject a free update on any cell where the same frame returned something in the obstacle band.
- Map coverage is thin. A 150-second run maps roughly 16 percent of the true obstacle cells. The robot commits to a 27 m route while sensing 8 m ahead, so most of any plan crosses territory it has never seen.
- Perception is the bottleneck. About 6.4 seconds per cycle on CPU, split as 2.9 seconds detecting and 3.6 seconds segmenting, rising to 14 seconds when many distinct objects are in frame. A CUDA build is the real answer.
- No recovery from inside a solid obstacle. The planner can escape its inflation ring, but if the robot is genuinely embedded in an obstacle footprint it stops rather than driving out.
- Terrain semantics are unused. Grass, gravel and mud are painted into the scene and named in the XML, but the prompt only asks for obstacles, so the traversal-cost tiers the scene was designed around are never exercised.
- The survey happens once. A goal that changes mid-run, or a corridor entered blind, gets no second look.
- `requirements.txt` is empty, and there is no license file.
- Roadmap: a fast geometric obstacle layer built from the depth image to complement the slow semantic one, speed capped by how far the map is known ahead, and persisting or visualising the costmap.

---

## License

No license has been chosen yet. There is no license file in this repository, which by default means all rights are reserved and others have no permission to use, copy or distribute the work. If you intend this to be open source, add a license file. MIT is the usual choice for a project like this.

---

## Contact

- Project maintainer: Lucas, lucas.vanwinckel@gmail.com
- Project link: https://github.com/lucasvw12/Semantic-Costmap-Navigation
