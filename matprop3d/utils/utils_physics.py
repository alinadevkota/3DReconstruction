import numpy as np
import warp as wp

def get_null_mass():
    """Get Null Mass and Null Interia"""
    I = np.zeros((3,3))
    iI = np.zeros((3,3))
    return 0.0, 0.0, I, iI
    

def sphere_inertia_tensor(mass, radius, hollow=False):
    """
    Calculate the inertia tensor of a sphere.

    Args:
        mass (float): Mass of the sphere.
        radius (float): Radius of the sphere.
        hollow (bool): True for a hollow sphere, False for a solid sphere.

    Returns:
        np.ndarray: Inertia tensor (3x3 matrix).
    """
    # Moment of inertia for a sphere about its diameter
    if hollow:
        I = (2/3) * mass * radius**2  # Hollow sphere
        invI = 1/I
    else:
        I = (2/5) * mass * radius**2  # Solid sphere
        invI = 1/I

    # Inertia tensor is diagonal for a sphere
    inertia_tensor = np.diag([I, I, I])
    inv_inertia_tensor = np.diag([invI, invI, invI])
    
    return mass, 1/mass, inertia_tensor, inv_inertia_tensor


if __name__=='__main__':
    m, im, I, iI = sphere_inertia_tensor(113.09735, 0.3)
    em, eim, eI, eiI = get_null_mass()

    wp_m = wp.array([m, em], dtype=float, requires_grad=True)
    wp_im = wp.array([im, eim], dtype=float, requires_grad=True)
    wp_I = wp.array([I, eI], dtype=wp.mat33f, requires_grad=True)
    wp_iI = wp.array([iI, eiI], dtype=wp.mat33f, requires_grad=True)

    print(wp_m)
    print(wp_im)
    print(wp_I)
    print(wp_iI)