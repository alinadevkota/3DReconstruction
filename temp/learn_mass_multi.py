import numpy as np

import warp as wp
import warp.sim
import warp.sim.render

from matprop3d.utils.env_utils import PoolEnvironment
from matprop3d.utils.sim_utils import update_camera
from calc_inertial import get_null_mass, sphere_inertia_tensor

from warp.sim import ModelBuilder
from tqdm import tqdm

   
#! Loss Kernels 
@wp.kernel
def loss_kernel_body(pos: wp.array(dtype=wp.transform), target: wp.vec3, loss: wp.array(dtype=float)):
    pose = pos[0]
    trans = wp.transform_get_translation(pose)
    delta = trans - target
    loss[0] = wp.dot(delta, delta)
    
@wp.kernel
def loss_kernel_body_1(pos: wp.array(dtype=wp.transform), target: wp.vec3, loss: wp.array(dtype=float)):
    pose = pos[1]
    trans = wp.transform_get_translation(pose)
    delta = trans - target
    loss[0] = wp.dot(delta, delta)
    
    
#! Optimization Kernels
@wp.kernel
def step_kernel_mass(x: wp.array(dtype=wp.float32), grad: wp.array(dtype=wp.float32), alpha: float):
    grad_item = grad[0] 
    
    #! Gradient clipping 
    if grad_item>100.0:
        grad_item = 100.0
    elif grad_item<-100.0:
        grad_item = -100.0
        
    x[0] = wp.abs(x[0] - grad_item * alpha)
    

#! Optimization Kernels
@wp.kernel
def step_kernel_mass_1(x: wp.array(dtype=wp.float32), grad: wp.array(dtype=wp.float32), alpha: float):
    grad_item = grad[1] 
    
    #! Gradient clipping 
    if grad_item>100.0:
        grad_item = 100.0
    elif grad_item<-100.0:
        grad_item = -100.0
        
    x[1] = wp.abs(x[1] - grad_item * alpha) 

    
@wp.kernel
def step_kernel_veloclity(
    x: wp.array(dtype=wp.spatial_vectorf), 
    grad: wp.array(dtype=wp.spatial_vectorf), 
    alpha: float
):
    tid = wp.tid()
    
    x[tid] = x[tid] - grad[tid] * alpha
    

class MultiBodyCollision:
    def __init__(self):
        #! Simulation Params
        sim_duration =  2.0
        fps = 60
        self.frame_dt = 1.0 / fps
        frame_steps = int(sim_duration / self.frame_dt)
        self.sim_substeps = 8
        self.sim_steps = frame_steps * self.sim_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.iter = 0
        self.render_time = 0.0


        #! Create Model
        builder = ModelBuilder()
        
        PoolEnvironment(builder)
        
        #! Add Initial Velocity to the Objects 
        builder.body_qd[0] = [0,0,0,-3,0,0.5]
        builder.body_qd[1] = [0,0,0,2,0,1]
           
        #! Finalize Model bulding 
        self.model = builder.finalize(requires_grad=True)
        self.model.ground = True

        #! Targets
        self.target_0 = (-2.037689,    0.45728338, -0.29389492)
        self.target_1 = (1.8676655,   0.34936282,  1.2584252)

        #! Integrator
        self.integrator = wp.sim.SemiImplicitIntegrator()
       
        #! Training Params 
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        self.loss_1 = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        self.train_rate_body = 1e-3


        # allocate sim states for trajectory
        self.states = []
        for _i in range(self.sim_steps + 1):
            self.states.append(self.model.state())

        # one-shot contact creation (valid if we're doing simple collision against a constant normal plane)
        wp.sim.collide(self.model, self.states[0])

        #! Renderer
        self.renderer = wp.sim.render.SimRendererOpenGL(model=self.model, path=None, scaling=1.0)
        update_camera(self.renderer, cam=(0,5,5))



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
        wp.launch(loss_kernel_body, dim=1, inputs=[self.states[-1].body_q, self.target_0, self.loss])
        wp.launch(loss_kernel_body_1, dim=1, inputs=[self.states[-1].body_q, self.target_1, self.loss_1])
       

    def step(self):
            self.tape = wp.Tape()
            with self.tape:
                self.forward()
  
            #! --- Update for Body 0 ---
            # Back Propagate
            self.tape.backward(self.loss)

            # Update inv mass
            inv_mass = self.model.body_inv_mass
            wp.launch(step_kernel_mass, dim=1, inputs=[inv_mass, inv_mass.grad, self.train_rate_body])
           
            # Get mass and inertia 
            inv_mass_item = inv_mass.numpy()[0]
            m, im, I, iI = sphere_inertia_tensor(1/inv_mass_item, 0.3)
            
            # Log it
            with open('mass.log', 'a') as fp:
                fp.write(f"{inv_mass_item}\n")

            #! --- Update for Body 1 ---
            # Back Propagate
            self.tape.zero()
            self.tape.backward(self.loss_1)
                
            # Update inv mass
            inv_mass = self.model.body_inv_mass
            wp.launch(step_kernel_mass_1, dim=1, inputs=[inv_mass, inv_mass.grad, self.train_rate_body])
           
            # Get mass and inertia 
            inv_mass_item_1 = inv_mass.numpy()[1]
            m1, im1, I1, iI1 = sphere_inertia_tensor(1/inv_mass_item_1, 0.2)
            
            # Log it
            with open('mass1.log', 'a') as fp:
                fp.write(f"{inv_mass_item_1}\n")


            #! Update model
            em, eim, eI, eiI = get_null_mass()
            self.model.body_mass = wp.array([m, m1, em], dtype=float, requires_grad=True)
            self.model.body_inv_mass = wp.array([im, im1, eim], dtype=float, requires_grad=True)
            self.model.body_inertia = wp.array([I, I1, eI], dtype=wp.mat33f, requires_grad=True)
            self.model.body_inv_inertia = wp.array([iI, iI1, eiI], dtype=wp.mat33f, requires_grad=True)
            

            
            #! Update density
            # print("Body stats:")
            # print(self.model.body_mass) 
            # print(self.model.body_inv_mass)
            # print(self.model.body_inertia)
            # print(self.model.body_inv_inertia)
            # print("sdfsdf")
            # exit()
            
            self.tape.zero()
            self.iter = self.iter + 1


    def render(self):
        if self.renderer is None:
            return


        traj_verts_body_0 = [self.states[0].body_q.numpy()[0][:3].tolist()]
        traj_verts_body_1 = [self.states[0].body_q.numpy()[1][:3].tolist()]
        
        
        for i in range(0, self.sim_steps, self.sim_substeps):
            #! Get Trajectories
            traj_verts_body_0.append(self.states[i].body_q.numpy()[0][:3].tolist())
            traj_verts_body_1.append(self.states[i].body_q.numpy()[1][:3].tolist())
            
            self.renderer.begin_frame(self.render_time)
            self.renderer.render(self.states[i])

            #! Render Targets
            self.renderer.render_box(
                pos=self.target_0,
                rot=wp.quat_identity(),
                extents=(0.1, 0.1, 0.1),
                name="target_0",
                color=(0.0, 1.0, 0.0),
            )
            
            self.renderer.render_box(
                pos=self.target_1,
                rot=wp.quat_identity(),
                extents=(0.1, 0.1, 0.1),
                name="target_1",
                color=(1.0, 0.0, 0.0),
            )
           
            #! Render Trajectory 
            self.renderer.render_line_strip(
                    vertices=traj_verts_body_0,
                    color=wp.render.bourke_color_map(0.0, 7.0, self.loss.numpy()[0]),
                    radius=0.02,
                    name=f"traj_body_0_{self.iter-1}",
            )

            self.renderer.render_line_strip(
                    vertices=traj_verts_body_1,
                    color=wp.render.bourke_color_map(0.0, 7.0, self.loss.numpy()[0]),
                    radius=0.02,
                    name=f"traj_body_1_{self.iter-1}",
            )
            
            self.renderer.end_frame()

            self.render_time += self.frame_dt


#! Program Starts Here
if __name__ == "__main__":
    device = 'cuda'
    train_iters = 250

    with wp.ScopedDevice(device):
        model = MultiBodyCollision()

        for i in tqdm(range(train_iters)):
            model.step()
            if i % 2 == 0:
                model.render()
