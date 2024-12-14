import cv2
import numpy as np
import warp as wp

# Example simulation data setup
num_objects = 10
# positions = wp.array([[i, i, 0.0] for i in range(num_objects)], dtype=wp.vec3)
positions = wp.array(
    [
        [
            np.random.uniform(-100, 100),
            np.random.uniform(-100, 100),
            np.random.uniform(-100, 100),
        ]
        for _ in range(num_objects)
    ],
    dtype=wp.vec3,
)
# velocities = wp.array([[0.1, 0.1, 0.1] for _ in range(num_objects)], dtype=wp.vec3)
velocities = wp.array(
    [
        [
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10),
        ]
        for _ in range(num_objects)
    ],
    dtype=wp.vec3,
)

# Simulation parameters
radius = 10
num_frames = 1000  # Number of frames to simulate
frame_size = (500, 500)  # Size of the video frame in pixels
fps = 30  # Frames per second for the video

# Video writer setup
output_filename = "simulation_output_with_collisions.avi"
fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Codec
out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)


@wp.kernel
def update_positions(
    positions: wp.array(dtype=wp.vec3), velocities: wp.array(dtype=wp.vec3), dt: float
):
    i = wp.tid()
    positions[i] += velocities[i] * dt


@wp.kernel
def detect_collisions(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    radius: float,
    num_objects: int,
):  # Pass the number of objects explicitly
    i = wp.tid()
    for j in range(num_objects):  # Use num_objects instead of len()
        if i != j:
            dir = positions[j] - positions[i]
            dist = wp.length(dir)
            if dist < 2.0 * radius:  # Cast `2` to a float by using `2.0`
                velocities[i] = -velocities[i]  # Reflect velocity


# Projection function: 3D to 2D conversion with camera position
def project_to_2d(position, camera_position, frame_size):
    # Translate the object position relative to the camera
    relative_position = position - camera_position

    # Apply a simple perspective projection (ignoring field of view, etc. for simplicity)
    # We use perspective scaling based on the z-coordinate (closer objects are larger)
    scale = 500 / (
        relative_position[2] + 500
    )  # Simple perspective scale factor based on z-distance

    # Project the 3D position to 2D by scaling the x, y coordinates
    x = int(relative_position[0] * scale + 250)
    y = int(relative_position[1] * scale + 250)

    # Return 2D position along with size factor (depth effect)
    return x, y, abs(scale)


# Camera parameters
camera_position = np.array([0, 0, 1000])  # Camera at (250, 250, 1000) in the 3D space
camera_focus = np.array(
    [250, 250, 0]
)  # Camera looking towards the center of the scene (origin)
camera_up = np.array([0, 1, 0])  # Camera up vector (Y axis)

# Render and save frames in a loop
for frame_idx in range(num_frames):
    # Simulate one step
    wp.launch(update_positions, dim=num_objects, inputs=[positions, velocities, 0.1])

    # Detect collisions between objects
    wp.launch(
        detect_collisions,
        dim=num_objects,
        inputs=[positions, velocities, radius, num_objects],
    )

    # Create a blank frame (black background)
    frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)

    positions_host = positions.numpy()
    # Draw particles as circles
    for pos in positions_host:
        print("pos", pos)
        x, y, size_factor = project_to_2d(pos, camera_position, frame_size)
        print(size_factor, x, y)
        cv2.circle(
            frame, (x, y), int(radius * size_factor), (0, 255, 0), -1
        )  # Draw circle with size based on depth

    # Write frame to video
    out.write(frame)

# Release the video writer
out.release()
print(f"Video saved as {output_filename}")
