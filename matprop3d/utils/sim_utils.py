from dataclasses import dataclass
import numpy as np

import warp as wp
from warp.sim import ModelBuilder

@dataclass
class Box():
    shape: str = 'box'
    trans: tuple[float, float, float] = (0,0,0)
    quat: tuple[float, float, float, float] = (0,0,0,1)
    dim: tuple[float, float, float] = (1.0,1.0,1.0)
    density: float = 0.0
    collision_group: int = -1
    
@dataclass
class Sphere():
    shape: str = 'sphere'
    trans: tuple[float, float, float] = (0,0,0)
    quat: tuple[float, float, float, float] = (0,0,0,1)
    radius: float = 0.0
    density: float = 0.0
    collision_group: int = -1


def create_body(
    builder:ModelBuilder, name: str, t:tuple[float], q:tuple[float], 
    body_parts: list[Box|Sphere]
)->None:
    """
    Args:
        builder (warp.sim.ModelBuilder)
        name (str): Name
        t (tuple(3)): Translation
        q (tuple(3)): Quaternion, Rotation
        body_parts (List[Box|Sphere]): List of body parts
    Returns:
        None
    """
    body = builder.add_body(
        origin=wp.transform(t, q),
        name=name
    )
    
    for bp in body_parts:
        if bp.shape =='box':
            builder.add_shape_box(
                body,
                pos=bp.trans,
                rot=bp.quat,
                hx=bp.dim[0]/2,
                hy=bp.dim[1]/2,
                hz=bp.dim[2]/2,
                density=bp.density,
                # collision_group=0, #! dont know
            )
            
        if bp.shape == 'sphere':
            builder.add_shape_sphere(
                body,
                radius = bp.radius,
                density=bp.density,
                # collision_group=bp.collision_group,
            )
            

def update_camera(renderer, cam=(5,5,5), target=(0,0,0)):
    camera_pos = np.array(cam)
    target_pos = np.array(target)
    camera_front = np.subtract(target_pos, camera_pos)  # `target_position`
    camera_front = camera_front / np.linalg.norm(camera_front)  # Normalize
    camera_up = (0, 1, 0)
    renderer.update_view_matrix(
        cam_pos=camera_pos, cam_front=camera_front, cam_up=camera_up
    )