import warp as wp
import warp.sim
import warp.sim.render

from warp.sim import ModelBuilder
from warp.sim.render import SimRendererOpenGL
from warp.sim import SemiImplicitIntegrator, XPBDIntegrator


#! matprop3d imports
from matprop3d.utils.env_utils import PoolEnvironment

#! Builds Simultation model
builder = ModelBuilder()


#! Create Playground
PoolEnvironment(builder)

print(builder.body_count)
# exit()


#! Add motions
x_vel = 50
y_vel = -10
builder.body_qd[4] = [0,0,0,x_vel,0,y_vel]


#! Finalize building
model = builder.finalize("cuda")
model.ground = True
state = model.state()

#! Integrator is like physics engine
integrator = SemiImplicitIntegrator()
# integrator = XPBDIntegrator()

#! Init Renderer
renderer = SimRendererOpenGL(
    model=model,
    path="output/",
    scaling=1.0,
)

#! Change camera pose
import numpy as np
camera_pos = np.array([5,5,0])
camera_front = np.subtract((0, 0, 0), camera_pos)  # `target_position`
camera_front = camera_front / np.linalg.norm(camera_front)  # Normalize
camera_up = (0, 1, 0)
renderer.update_view_matrix(
    cam_pos=camera_pos, cam_front=camera_front, cam_up=camera_up
)

fps = 30.0
sim_substeps = 8
dt = 1/fps
sim_dt = dt/sim_substeps

state1, state2 = None, None
state1 = state


#! Simulation begins here
for i in range(10000):
    state2=state1
    wp.sim.collide(model, state)
   
    state.clear_forces()
    integrator.simulate(model, state1, state2, dt=sim_dt)
   
    # Render the frame
    renderer.begin_frame(i * sim_dt)
    renderer.render(state1)
    renderer.end_frame()