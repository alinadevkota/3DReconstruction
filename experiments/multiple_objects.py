import warp as wp

# Define properties for multiple objects
radius = 0.5
num_objects = 100
positions = wp.array([[0.0, 0.0, 0.0] for _ in range(num_objects)], dtype=wp.vec3)
velocities = wp.array([[0.0, 0.0, 0.0] for _ in range(num_objects)], dtype=wp.vec3)
masses = wp.array([1.0 for _ in range(num_objects)], dtype=float)


@wp.kernel
def apply_gravity(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=float),
    dt: float,
):
    tid = wp.tid()
    if masses[tid] > 0.0:  # Only update objects with mass
        g = wp.vec3(0.0, -9.8, 0.0)  # Gravity vector
        velocities[tid] += g * dt
        positions[tid] += velocities[tid] * dt


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


dt = 0.01  # Time step
# Launch the kernel with all required inputs
wp.launch(
    detect_collisions,
    dim=num_objects,
    inputs=[positions, velocities, radius, num_objects],
)

for _ in range(1000):  # 100 simulation steps
    wp.launch(
        apply_gravity, dim=num_objects, inputs=[positions, velocities, masses, dt]
    )
    wp.launch(
        detect_collisions,
        dim=num_objects,
        inputs=[positions, velocities, radius, num_objects],
    )
