
import warp as wp
import warp.sim
import warp.sim.render

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

   
#! Optimization kernerl using adam 
@wp.kernel
def step_kernel_veloclity(
    x: wp.array(dtype=wp.spatial_vectorf), 
    grad: wp.array(dtype=wp.spatial_vectorf), 
    alpha: float
):
    tid = wp.tid()
    x[tid] = x[tid] - grad[tid] * alpha
    
    
@wp.kernel
def adam_step_kernel_float_0(
    g: wp.array(dtype=float),
    m: wp.array(dtype=float),
    v: wp.array(dtype=float),
    lr: float,
    beta1: float,
    beta2: float,
    t: float,
    eps: float,
    params: wp.array(dtype=float),
):
    i = 0
    m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
    v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i]
    mhat = m[i] / (1.0 - wp.pow(beta1, (t + 1.0)))
    vhat = v[i] / (1.0 - wp.pow(beta2, (t + 1.0)))
    params[i] = params[i] - lr * mhat / (wp.sqrt(vhat) + eps)
    
@wp.kernel
def adam_step_kernel_float_1(
    g: wp.array(dtype=float),
    m: wp.array(dtype=float),
    v: wp.array(dtype=float),
    lr: float,
    beta1: float,
    beta2: float,
    t: float,
    eps: float,
    params: wp.array(dtype=float),
):
    i = 1
    m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
    v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i]
    mhat = m[i] / (1.0 - wp.pow(beta1, (t + 1.0)))
    vhat = v[i] / (1.0 - wp.pow(beta2, (t + 1.0)))
    params[i] = params[i] - lr * mhat / (wp.sqrt(vhat) + eps)
    