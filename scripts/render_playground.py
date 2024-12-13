import warp as wp
import warp.sim
import warp.sim.render

from warp.sim import ModelBuilder
from warp.sim.render import SimRendererOpenGL
from warp.sim import SemiImplicitIntegrator, XPBDIntegrator
import numpy as np
np.set_printoptions(suppress=True, precision=3)

#! matprop3d imports
from matprop3d.utils.utils_env import PoolEnvironment
from matprop3d.utils.utils_sim import update_camera

if __name__=='__main__':
    #! Builds Simultation model
    builder = ModelBuilder()

    #! Create Playground
    ball_properties = [
    #   [Radius, Density, (x,y)]
        [0.3, 300, (1.9,-1)],
        [0.32, 100, (-1,-1)],
        [0.2, 150, (1,0.6)],
        [0.31, 100, (0.8,-0.6)]
    ]
    PoolEnvironment(builder, ball_properties)

    #! Add motions
    builder.body_qd[0] = [0,0,0,2,0,-1]
    builder.body_qd[1] = [0,0,0,-5,0,2]
    builder.body_qd[2] = [0,0,0,1,0,-3]
    builder.body_qd[3] = [0,0,0,7,0,4]


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
        path="Pool Environment Simulation",
        scaling=1.0,
    )

    #! Change camera pose
    update_camera(renderer, cam=(5,5,0))

    fps = 30.0
    sim_substeps = 8
    dt = 1/fps
    sim_dt = dt/sim_substeps

    state1, state2 = None, None
    state1 = state

    simulation_steps = 3000
    #! Simulation begins here
    for i in range(simulation_steps):
        state2=state1
        wp.sim.collide(model, state)
    
        state.clear_forces()
        integrator.simulate(model, state1, state2, dt=sim_dt)
        
        #! Render the frame
        renderer.begin_frame(i * sim_dt)
        renderer.render(state1)
        renderer.end_frame()
        
    print("Simulation Completed.")