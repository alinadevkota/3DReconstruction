import cv2
import numpy as np
import warp as wp

# Example simulation data setup
num_objects = 10
positions = wp.array([[i, i, 0.0] for i in range(num_objects)], dtype=wp.vec3)
velocities = wp.array([[0.1, 0.1, 0.1] for _ in range(num_objects)], dtype=wp.vec3)

# Simulation parameters
radius = 0.2
num_frames = 100  # Number of frames to simulate
frame_size = (500, 500)  # Size of the video frame in pixels
fps = 30  # Frames per second for the video

# Video writer setup
output_filename = "simulation_output.avi"
fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Codec
out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)


@wp.kernel
def update_positions(
    positions: wp.array(dtype=wp.vec3), velocities: wp.array(dtype=wp.vec3), dt: float
):
    i = wp.tid()
    positions[i] += velocities[i] * dt


# Render and save frames in a loop
for frame_idx in range(num_frames):
    # Simulate one step
    wp.launch(update_positions, dim=num_objects, inputs=[positions, velocities, 0.1])

    # Create a blank frame (black background)
    frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)

    positions_host = positions.numpy()
    # Draw particles as circles
    for pos in positions_host:
        x = int(pos[0] * 50 + frame_size[0] / 2)  # Scale and translate
        y = int(pos[1] * 50 + frame_size[1] / 2)
        cv2.circle(frame, (x, y), int(radius * 50), (0, 255, 0), -1)  # Draw circle

    # Write frame to video
    out.write(frame)

# Release the video writer
out.release()
print(f"Video saved as {output_filename}")
