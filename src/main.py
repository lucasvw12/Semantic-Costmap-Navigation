from simulator import Simulator
from mapping import OccupancyGrid
from mapping import Mapping
#from raytracing import RayTracer
rgb_image = None
depth_image = None
def main():
    simulator = Simulator()
    simulator.run()

if __name__ == "__main__":
    print("Starting simulation...")
    main()
 