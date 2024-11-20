import cv2
import numpy as np
import warp as wp
import warp.render  # noqa: E501

from .utils import get_view_matrix

np.random.seed(0)


# Warp kernel to update particle positions
@wp.kernel
def update_positions(
    positions: wp.array(dtype=wp.vec3), velocities: wp.array(dtype=wp.vec3), dt: float
):
    i = wp.tid()
    positions[i] += velocities[i] * dt


# Warp kernel to detect particle collisions
@wp.kernel
def detect_collisions(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    radius: float,
    num_objects: int,
):
    i = wp.tid()
    for j in range(num_objects):
        if i != j:
            dir = positions[j] - positions[i]
            dist = wp.length(dir)
            if dist < 2.0 * radius:
                velocities[i] = -velocities[i]  # Reflect velocity


# Simulation parameters
num_objects = 10

# Set up Warp's OpenGLRenderer
renderer = wp.render.OpenGLRenderer(vsync=False, headless=False)
tile_indices = list(range(num_objects + 1))
# renderer.setup_tiled_rendering([tile_indices], projection_matrices=[[1,0,0,0,1,0,0,0,1]], view_matrices=[[1,0,0,0,1,0,0,0,1]]) # noqa: E501

C = np.array([0.0, 2.0, 10.0])  # Camera position
T = np.array([0.0, 0.0, 0.0])  # Target position (where the camera is looking)
U = np.array([0.0, 1.0, 0.0])  # Up vector

my_view_matrix = get_view_matrix(C, T, U)

renderer._view_matrix = np.array(my_view_matrix)


renderer.setup_tiled_rendering(
    [tile_indices]
)  # , view_matrices=[[1,0,0,0, 0,1,0,5, 0,0,1,0, 0,0,0,1]])

renderer.render_ground()

renderer.camera_fov = 25.0
renderer.camera_near_plane = 0.1
renderer.camera_far_plane = 100
renderer.update_projection_matrix()

print("tile matrices")
print(np.array(renderer._view_matrix))

# exit()

# print(renderer._tile_projection_matrices[0].reshape(4,4))

# positions = np.random.uniform(-1, 1, (num_objects, 3))
# velocities = np.random.uniform(-0.1, 0.1, (num_objects, 3))

# # Convert to wp arrays of dtype wp.vec3
# positions = wp.array(positions, dtype=wp.vec3)
# velocities = wp.array(velocities, dtype=wp.vec3)

positions = wp.array(
    [
        [np.random.uniform(-10, 10), 6, np.random.uniform(-10, 10)]
        for _ in range(num_objects)
    ],
    dtype=wp.vec3,
)
# velocities = wp.array([[0.1, 0.1, 0.1] for _ in range(num_objects)], dtype=wp.vec3)
velocities = wp.array(
    [
        [np.random.uniform(-1, 1), 0, np.random.uniform(-1, 1)]
        for _ in range(num_objects)
    ],
    dtype=wp.vec3,
)


radius = 0.1  # Adjust radius of the 3D particles
num_frames = 1000  # Number of frames to simulate
fps = 30  # Frames per second for the video
output_filename = "simulation_3d_output.avi"
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(
    output_filename, fourcc, fps, (renderer.screen_width, renderer.screen_height)
)

pixel_shape = (1, renderer.screen_height, renderer.screen_width, 3)

# Simulation and rendering loop
for frame_idx in range(num_frames):
    # Simulate one step
    wp.launch(update_positions, dim=num_objects, inputs=[positions, velocities, 0.1])
    wp.launch(
        detect_collisions,
        dim=num_objects,
        inputs=[positions, velocities, radius, num_objects],
    )

    # Begin frame rendering
    renderer.begin_frame(renderer.clock_time)

    # Draw particles in 3D space
    positions_host = positions.numpy()

    print(f"Frame {frame_idx}:")
    print(f"Last Sphere Position: {positions_host[0]}")
    print(f"Last Sphere Velocity: {velocities.numpy()[0]}")

    for i, pos in enumerate(positions_host):
        renderer.render_sphere(
            pos=tuple(pos),  # Current particle position (x, y, z)
            rot=(0, 0, 0, 1.0),  # No rotation for simplicity
            radius=radius,  # Particle radius
            color=[1.0, 0.0, 0.0],  # Red color
            name=f"particle{i}",
        )
    try:
        renderer.end_frame()
    except RuntimeError as e:
        print("Renderer failed:", e)
        print("Positions:", positions_host)
        print("Velocity:", velocities)
        raise

    pixels = wp.zeros(pixel_shape, dtype=wp.float32)
    # Capture the frame for the video
    renderer.get_pixels(pixels, mode="rgb")  # Get pixel data from renderer
    frame = pixels.numpy()[0]
    frame = (frame * 255).astype(np.uint8)
    frame = np.flip(frame, axis=0)  # Flip to match OpenGL coordinate system
    frame_to_write = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Write the frame to the video file
    out.write(frame_to_write)


# Release the video writer and cleanup
out.release()

renderer.clear()
print(f"3D simulation video saved as {output_filename}")
