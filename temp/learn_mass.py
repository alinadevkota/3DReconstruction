import numpy as np

import warp as wp
import warp.sim
import warp.sim.render

from matprop3d.utils.env_utils import PoolEnvironment
from matprop3d.utils.sim_utils import update_camera
from calc_inertial import get_null_mass, sphere_inertia_tensor

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
def multi_loss_kernel_body(
    pred0: wp.array(dtype=wp.transform), 
    pred1: wp.array(dtype=wp.transform), 
    pred2: wp.array(dtype=wp.transform), 
    pred3: wp.array(dtype=wp.transform), 
    pred4: wp.array(dtype=wp.transform), 
    
    target0: wp.vec3, 
    target1: wp.vec3, 
    target2: wp.vec3, 
    target3: wp.vec3, 
    target4: wp.vec3, 
    
    loss: wp.array(dtype=float)
):
    pose0 = pred0[0]
    trans0 = wp.transform_get_translation(pose0)
    delta0 = trans0 - target0
    loss0 = wp.dot(delta0, delta0)
    
    pose1 = pred1[0]
    trans1 = wp.transform_get_translation(pose1)
    delta1 = trans1 - target1
    loss1 = wp.dot(delta1, delta1)
    
    pose2 = pred2[0]
    trans2 = wp.transform_get_translation(pose2)
    delta2 = trans2 - target2
    loss2 = wp.dot(delta2, delta2)
    
    pose3 = pred3[0]
    trans3 = wp.transform_get_translation(pose3)
    delta3 = trans3 - target3
    loss3 = wp.dot(delta3, delta3)
    
    pose4 = pred4[0]
    trans4 = wp.transform_get_translation(pose4)
    delta4 = trans4 - target4
    loss4 = wp.dot(delta4, delta4)
    
    # loss[0] = loss0+loss1+loss2+loss3+loss4
    loss[0] = loss0+loss1+loss3+loss4


@wp.kernel
def step_kernel_mass(x: wp.array(dtype=wp.float32), grad: wp.array(dtype=wp.float32), alpha: float):
    tid = wp.tid()
    x[tid] = wp.abs(x[tid] - grad[tid] * alpha)
    
    
@wp.kernel
def step_kernel(x: wp.array(dtype=wp.transform), grad: wp.array(dtype=wp.transform), alpha: float):
    tid = wp.tid()
    
    trans = wp.transform_get_translation(x[tid])
    trans_grad = wp.transform_get_translation(grad[tid])

    # gradient descent step
    trans_updated = trans - trans_grad * alpha

    quat = wp.transform_get_rotation(x[tid])
    
    x[tid] = wp.transform(trans_updated, quat)
    
@wp.kernel
def step_kernel_veloclity(
    x: wp.array(dtype=wp.spatial_vectorf), 
    grad: wp.array(dtype=wp.spatial_vectorf), 
    alpha: float
):
    tid = wp.tid()
    
    x[tid] = x[tid] - grad[tid] * alpha
    

class Example:
    def __init__(self):
        #! Simulation Params
        sim_duration =  1.4
        fps = 60
        self.frame_dt = 1.0 / fps
        frame_steps = int(sim_duration / self.frame_dt)

        # sim frequency
        self.sim_substeps = 10
        self.sim_steps = frame_steps * self.sim_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.iter = 0
        self.render_time = 0.0


        #! Create Model
        builder = ModelBuilder()
        
        PoolEnvironment(builder)
        
        
        builder.body_qd[0] = [0,0,0,-3,0,0.5]
        # builder.body_qd[0] = [5,0,5,0,0,0]
           
        #! Finalize Model bulding 
        self.model = builder.finalize(requires_grad=True)
        self.model.ground = True


        #! Integrator
        self.integrator = wp.sim.SemiImplicitIntegrator()

        #! Training Params
        # self.target2 = (-1.0279006,   0.4391355,   0.)
        self.target_200 = (-9.6696705e-01,  4.4891089e-01,  1.6116117e-01)
        self.target_400 = (-1.85306621e+00,  4.48910892e-01,  3.08844447e-01)
        self.target_600 = (-1.9243586,   0.45005232,  0.4321407)
        self.target_800 = (-1.6362988,   0.44891092,  0.5480492)
        
        self.target2 = (-1.4070804,   0.44891092,  0.6402814)
        
        self.target3 = (-1.1499655e+00,  4.4891092e-01, -4.9167269e-01)
        
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        self.train_rate_body = 1e-1


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
            # if i%200==0:
                # print('Trajectory Print:', i)
                # print(self.states[i].body_q)
            
        # wp.launch(
        #     multi_loss_kernel_body,
        #     dim=1,
        #     inputs=[
        #         self.states[200].body_q,
        #         self.states[400].body_q,
        #         self.states[600].body_q,
        #         self.states[800].body_q,
        #         self.states[-1].body_q,
                
        #         self.target_200,
        #         self.target_400,
        #         self.target_600,
        #         self.target_800,
        #         self.target2,
                
        #         self.loss
        #     ]
        # )

        wp.launch(loss_kernel_body, dim=1, inputs=[self.states[-1].body_q, self.target3, self.loss])
       
        # print("forward function") 
        # print(self.states[-1].body_q)
        # print("forward function ends")
        # exit()

        return self.loss

    def step(self):
            self.tape = wp.Tape()
            with self.tape:
                self.forward()
            self.tape.backward(self.loss)

  
            #! Update initial velocity of body
            # print("updating intial velocity")
            # x_body = self.states[0].body_qd

            # print(x_body)
            # print(x_body.dtype)
            
            # print(x_body.grad)
            # wp.launch(step_kernel, dim=len(x_body), inputs=[x_body, x_body.grad, self.train_rate_body])
            # wp.launch(step_kernel_veloclity, dim=1, inputs=[x_body, x_body.grad, self.train_rate_body])
            # x_body_grad = self.tape.gradients[self.states[0].body_qd]
            
            # exit()
            #! Update mass of body
            # print('updating body mass')
            x_body_mass = self.model.body_inv_mass
           
            print('Before Updating:','Mass:', x_body_mass, 'Mass Gradient:', x_body_mass.grad)
            
            # wp.launch(step_kernel_mass, dim=len(x_body_mass), inputs=[x_body_mass, x_body_mass.grad, self.train_rate_body])
            wp.launch(step_kernel_mass, dim=1, inputs=[x_body_mass, x_body_mass.grad, self.train_rate_body])
            self.model.body_inv_mass = x_body_mass
            
            print("-----------> mass: ", x_body_mass)
           
            inv_mass_learned = x_body_mass.numpy()[0]
            print(inv_mass_learned)

            m, im, I, iI = sphere_inertia_tensor(1/inv_mass_learned, 0.3)
            em, eim, eI, eiI = get_null_mass()

            self.model.body_mass = wp.array([m, em], dtype=float, requires_grad=True)
            self.model.body_inv_mass = wp.array([im, eim], dtype=float, requires_grad=True)
            self.model.body_inertia = wp.array([I, eI], dtype=wp.mat33f, requires_grad=True)
            self.model.body_inv_inertia = wp.array([iI, eiI], dtype=wp.mat33f, requires_grad=True)
            

            
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

        traj_verts_body = [self.states[0].body_q.numpy()[0][:3].tolist()]
        
        
        for i in range(0, self.sim_steps, self.sim_substeps):
            traj_verts_body.append(self.states[i].body_q.numpy()[0][:3].tolist())
            
            self.renderer.begin_frame(self.render_time)
            self.renderer.render(self.states[i])

            self.renderer.render_box(
                pos=self.target3,
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
            if i % 2 == 0:
                # print("rendering now")
                example.render()

        # example.tape.visualize("bounce.dot")
        # print("saved ")