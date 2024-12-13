import numpy as np
from warp.sim import ModelBuilder
from tqdm import tqdm
import os
from pathlib import Path

import warp as wp
import warp.sim
import warp.sim.render

from matprop3d.utils.utils_sim import update_camera
from matprop3d.utils.utils_physics import get_null_mass, sphere_inertia_tensor
from matprop3d.utils.utils_warp import (
    loss_kernel,
    adam_step_kernel_float_0
)


class LearnDensitySim:
    def __init__(self):
        #! Path
        self.ws = Path(os.getenv('MATPROP3DWS'))
        
        #! Simulation Params
        sim_duration =  0.7
        fps = 120
        self.frame_dt = 1.0 / fps
        frame_steps = int(sim_duration / self.frame_dt)
        self.sim_substeps = 8
        self.sim_steps = frame_steps * self.sim_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.iter = 0
        self.render_time = 0.0

        #! Create Model
        drop_height = 0.7
        builder = ModelBuilder()
        builder.add_particle(
            pos=(0, drop_height, 0.0),
            vel=(0.0, 0.0, 0.0),
            mass=0.1,
            radius=0.1
        )
 
        #! Finalize Model bulding 
        self.model = builder.finalize(requires_grad=True)
        self.model.ground = True

        #! Targets
        rf = np.loadtxt(self.ws/'rebounce_factor.txt')
        self.target = (0.0, rf*drop_height, 0.0)
        
        #! Integrator
        self.integrator = wp.sim.SemiImplicitIntegrator()
       
        #! Training Params 
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        self.train_rate_body = 1e-1
        self.first_moment = wp.array([0.0], dtype=float)
        self.second_moment = wp.array([0.0], dtype=float)

        #! allocate sim states for trajectory
        self.states = []
        for _i in range(self.sim_steps + 1):
            self.states.append(self.model.state())
            
        # print(type(self.states[-1].particle_q))
        # print(self.states[-1].particle_q.dtype)
        # exit()
            
        #! Collision
        wp.sim.collide(self.model, self.states[0])

        #! Renderer
        self.renderer = wp.sim.render.SimRendererOpenGL(model=self.model, path=None, scaling=1.0)
        update_camera(self.renderer, cam=(0,0.6,3))


    def forward(self):
        #! Forward Pass (Simulation Steps)
        for i in range(self.sim_steps):
            wp.sim.collide(self.model, self.states[i])
            self.states[i].clear_forces()
            self.integrator.simulate(self.model, self.states[i], self.states[i + 1], self.sim_dt)
           
        #! Print Final Body Position after simulation 
        # print("Final Body Positions")
        # print(self.states[-1].body_q)
        # exit()

        #! Calculate Loss
        wp.launch(loss_kernel, dim=1, inputs=[self.states[-1].particle_q, self.target, self.loss])
        
        # loss_to_save = self.loss.numpy()
        # with open('file.txt', 'a') as fp:
        #     fp.write(str(loss_to_save[0]))
        #     fp.write('\n')


    def step(self):
        self.tape = wp.Tape()
        with self.tape:
            self.forward()

        # Back Propagate
        self.tape.backward(self.loss)

        # Update inv mass
        inv_mass = self.model.particle_inv_mass
        # wp.launch(step_kernel_mass, dim=1, inputs=[inv_mass, inv_mass.grad, self.train_rate_body])
        
        wp.launch(
            adam_step_kernel_float_0,
            dim=1,
            inputs=[
                    inv_mass.grad,
                    self.first_moment,
                    self.second_moment,
                    self.train_rate_body,
                    0.9,
                    0.99,
                    self.iter+1,
                    1e-8,
                    inv_mass
            ] 
        )
        
        # Get mass and inertia 
        self.model.particle_inv_mass = inv_mass
        self.model.particle_mass = 1/inv_mass.numpy() 
        # print(self.model.particle_mass)
        
        to_save = inv_mass.numpy()
        with open(self.ws/'real_inv_mass_learning.txt', 'a') as fp:
            fp.write(str(to_save[0]))
            fp.write('\n')

        self.tape.zero()
        self.iter = self.iter + 1


    def render(self):
        if self.renderer is None:
            return

        traj_verts_body_0 = [self.states[0].particle_q.numpy()[0][:3].tolist()]
        # traj_verts_body_1 = [self.states[0].body_q.numpy()[1][:3].tolist()]
        
        for i in range(0, self.sim_steps, self.sim_substeps):
            #! Get Trajectories
            traj_verts_body_0.append(self.states[i].particle_q.numpy()[0][:3].tolist())
            # traj_verts_body_1.append(self.states[i].body_q.numpy()[1][:3].tolist())
            
            self.renderer.begin_frame(self.render_time)
            self.renderer.render(self.states[i])

            #! Render Targets
            self.renderer.render_box(
                pos=self.target,
                rot=wp.quat_identity(),
                extents=(0.1, 0.01, 0.1),
                name="target_0",
                color=(0.0, 1.0, 0.0),
            )
            
           
            #! Render Trajectory 
            self.renderer.render_line_strip(
                    vertices=traj_verts_body_0,
                    color=wp.render.bourke_color_map(0.0, 7.0, self.loss.numpy()[0]),
                    radius=0.02,
                    name=f"traj_body_0_{self.iter-1}",
            )

            
            self.renderer.end_frame()

            self.render_time += self.frame_dt


if __name__ == "__main__":
    device = 'cuda'
    train_iters = 250
    import matplotlib.pyplot as plt
    from matprop3d.utils.utils_plot import plot_sphere_density_trajectory

    with wp.ScopedDevice(device):
        model = LearnDensitySim()

        for i in tqdm(range(train_iters)):
            model.step()
            if i % 2 == 0:
                model.render()
                
    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(5,5))
    ws = Path(os.getenv('MATPROP3DWS'))
    plot_sphere_density_trajectory(ax, ws/'real_inv_mass_learning.txt', radius=0.0335, gt=None, name='Real Ball')
    plt.savefig(ws/'density_estimation_real.png')