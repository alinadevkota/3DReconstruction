import matplotlib.pyplot as plt
import numpy as np
import cv2


def plot_bounding_boxes(image, bounding_boxes, color=(0,0,255)):
    """
    image (np.ndarray): Image
    bounding_boxes(np.ndarray or List[List[int]])
    """
    for bb in bounding_boxes:
        xmin, ymin, xmax, ymax = bb
        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), color, 10)
    return image


def plot_sphere_density_trajectory(ax, traj_file, radius, gt, name):
    inv_mass = np.loadtxt(traj_file)
    mass = 1/inv_mass
    vol = (4/3)*np.pi*radius*radius*radius
    density = mass/vol
   
    ax.axhline(y=gt, color='g', linestyle='--', label='GT Density')
    ax.set_yscale('log')
    ax.set_xlabel('Training Iterations')
    ax.set_ylabel('Density (kg/m3)')
    ax.set_title(name)
    ax.plot(density)
    ax.legend()
    
    
if __name__=='__main__':
    import os
    from pathlib import Path
    ws = Path(os.getenv('MATPROP3DWS'))
    
    fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(10,5))
    
    plot_sphere_density_trajectory(ax[0], ws/'inv_mass_0.log', radius=0.3, gt=185, name='Ball1')
    plot_sphere_density_trajectory(ax[1], ws/'inv_mass_1.log', radius=0.2, gt=200, name='Ball2')
    
    plt.savefig(ws/'density_estimation.png')
    # plt.show()

    
