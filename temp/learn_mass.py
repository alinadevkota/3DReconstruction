import numpy as np

import warp as wp
import warp.sim
import warp.sim.render

from matprop3d.utils.env_utils import PoolEnvironment
from matprop3d.utils.sim_utils import update_camera

from warp.sim import ModelBuilder
from tqdm import tqdm

@wp.kernel
def loss_kernel(pos: wp.array(dtype=wp.vec3), target: wp.vec3, loss: wp.array(dtype=float)):
    delta = pos[0] - target
    loss[0] = wp.dot(delta, delta)
    
@wp.kernel
def loss_kernel_body(pos: wp.array(dtype=wp.transform), target: wp.vec3, loss: wp.array(dtype=float)):
    pose = pos[0]
    trans = wp.transform_get_translation(pose)
    delta = trans - target
    loss[0] = wp.dot(delta, delta)


@wp.kernel
def step_kernel_mass(x: wp.array(dtype=wp.float32), grad: wp.array(dtype=wp.float32), alpha: float):
    tid = wp.tid()
    x[tid] = x[tid] - grad[tid] * alpha
    
    
@wp.kernel
def step_kernel(x: wp.array(dtype=wp.transform), grad: wp.array(dtype=wp.transform), alpha: float):
    tid = wp.tid()
    
    trans = wp.transform_get_translation(x[tid])
    trans_grad = wp.transform_get_translation(grad[tid])

    # gradient descent step
    trans_updated = trans - trans_grad * alpha

    quat = wp.transform_get_rotation(x[tid])
    
    x[tid] = wp.transform(trans_updated, quat)


class Example:
    def __init__(self):
        #! Simulation Params
        sim_duration =  0.6
        fps = 60
        self.frame_dt = 1.0 / fps
        frame_steps = int(sim_duration / self.frame_dt)

        # sim frequency
        self.sim_substeps = 8
        self.sim_steps = frame_steps * self.sim_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.iter = 0
        self.render_time = 0.0


        #! Create Model
        builder = ModelBuilder()
        
        PoolEnvironment(builder)
        
        
        #! Add a body Sphere
        # body = builder.add_body(
        #     origin=wp.transform((0.0, 1.0, 0.0), (0,0,0,1)),
        #     name="body_sphere"
        # ) 
        
        # builder.add_shape_sphere(
        #     body,
        #     radius = 0.1,
        #     density=500,
        #     collision_group=0,
        # )
        
        #! Add bouncing surface
        # body2 = builder.add_body(
        #     origin=wp.transform((0.0, 0.2, 0.0), (0,0,0,1)),
        #     name="bouncing surface"
        # ) 
        
        # builder.add_shape_box(
        #         body2,
        #         hx=2.0,
        #         hy=0.05,
        #         hz=2.0,
        #         density=0,
        #         collision_group=0, #! dont know
        #     )
       


        builder.body_qd[0] = [0,0,0,-3,0,0]
           
        #! Finalize Model bulding 
        self.model = builder.finalize(requires_grad=True)
        self.model.ground = True


        #! Integrator
        self.integrator = wp.sim.SemiImplicitIntegrator()

        #! Training Params
        self.target2 = (-1.0279006,   0.4391355,   0.)
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        self.train_rate_body = 0.01


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
        for i in range(self.sim_steps):
            wp.sim.collide(self.model, self.states[i])
            self.states[i].clear_forces()
            self.integrator.simulate(self.model, self.states[i], self.states[i + 1], self.sim_dt)


        wp.launch(loss_kernel_body, dim=1, inputs=[self.states[-1].body_q, self.target2, self.loss])
       
        print("forward function") 
        print(self.states[-1].body_q)
        print("forward function ends")
        # exit()

        return self.loss

    def step(self):
            self.tape = wp.Tape()
            with self.tape:
                self.forward()
            self.tape.backward(self.loss)

  
            #! Update initial velocity of body
            # print("updating intial pose")
            # x_body = self.states[0].body_q
            # print(x_body.grad)
            # wp.launch(step_kernel, dim=len(x_body), inputs=[x_body, x_body.grad, self.train_rate_body])
            # x_body_grad = self.tape.gradients[self.states[0].body_qd]
            
            #! Update mass of body
            print('updating body mass')
            x_body_mass = self.model.body_inv_mass
            print(x_body_mass, x_body_mass.grad)
            wp.launch(step_kernel_mass, dim=len(x_body_mass), inputs=[x_body_mass, x_body_mass.grad, self.train_rate_body])
            self.model.body_inv_mass = x_body_mass
            
            self.tape.zero()
            self.iter = self.iter + 1


    def render(self):
        if self.renderer is None:
            return

        traj_verts_body = [self.states[0].body_q.numpy()[0][:3].tolist()]
        
        
        for i in range(0, self.sim_steps, self.sim_substeps):
            traj_verts_body.append(self.states[i].body_q.numpy()[0][:3].tolist())
            
            self.renderer.begin_frame(self.render_time)
            self.renderer.render(self.states[i])

            self.renderer.render_box(
                pos=self.target2,
                rot=wp.quat_identity(),
                extents=(0.1, 0.1, 0.1),
                name="target2",
                color=(0.0, 1.0, 0.0),
            )
            
            self.renderer.render_line_strip(
                    vertices=traj_verts_body,
                    color=wp.render.bourke_color_map(0.0, 7.0, self.loss.numpy()[0]),
                    radius=0.02,
                    name=f"traj_body_{self.iter-1}",
            )
            
            self.renderer.end_frame()

            self.render_time += self.frame_dt


#! Program Starts Here
if __name__ == "__main__":
    device = None
    train_iters = 250

    with wp.ScopedDevice(device):
        example = Example()

        for i in tqdm(range(train_iters)):
            example.step()
            if i % 16 == 0:
                # print("rendering now")
                example.render()

        # example.tape.visualize("bounce.dot")
        # print("saved ")