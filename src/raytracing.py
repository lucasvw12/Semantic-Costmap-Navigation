# import mujoco
# import numpy as np


# class RayTracer:

#     def __init__(self, model, data, bodyexclude=-1):
#         self.model = model
#         self.data = data
#         self.bodyexclude = bodyexclude

#     def cast_rays(self, origin, directions, max_range=np.inf):

#         origin = np.ascontiguousarray(origin, dtype=np.float64).reshape(3)
#         directions = np.ascontiguousarray(directions, dtype=np.float64).reshape(-1, 3)

#         norms = np.linalg.norm(directions, axis=1, keepdims=True)
#         norms[norms == 0.0] = 1.0
#         directions = directions / norms

#         n_rays = len(directions)
#         geom_ids = np.full(n_rays, -1, dtype=np.int32)
#         distances = np.full(n_rays, -1.0, dtype=np.float64)

#         if n_rays:
#             cutoff = max_range if max_range > 0 else np.inf
#             mujoco.mj_multiRay(
#                 self.model,
#                 self.data,
#                 pnt=origin,
#                 vec=directions.ravel(),
#                 geomgroup=None,
#                 flg_static=1,
#                 bodyexclude=self.bodyexclude,
#                 geomid=geom_ids,
#                 dist=distances,
#                 normal=None,
#                 nray=n_rays,
#                 cutoff=cutoff,
#             )

#         points = np.full((n_rays, 3), np.nan, dtype=np.float64)
#         hit = geom_ids >= 0
#         points[hit] = origin + distances[hit, None] * directions[hit]

#         return points, distances, geom_ids

#     def cast_ray(self, origin, direction, max_range=np.inf):
#         points, distances, geom_ids = self.cast_rays(origin, [direction], max_range)
#         return points[0], float(distances[0]), int(geom_ids[0])
