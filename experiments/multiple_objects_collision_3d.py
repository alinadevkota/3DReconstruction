import math

import cv2
import numpy as np
import pygame
import warp as wp

# from OpenGL.GL import *
# from OpenGL.GLU import *
# from OpenGL.GLUT import *
# from pygame.locals import *
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_QUAD_STRIP,
    glBegin,
    glClear,
    glEnd,
    glPopMatrix,
    glPushMatrix,
    glTranslatef,
    glVertex3f,
)
from OpenGL.GLU import gluPerspective
from pygame.locals import DOUBLEBUF, OPENGL

np.random.seed(0)


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


# Function to generate a sphere
def draw_sphere(radius, slices=10, stacks=10):
    # Generate vertices and indices for a sphere
    for i in range(slices):
        lat0 = math.pi * (-0.5 + float(i) / slices)  # Latitude 0
        z0 = radius * math.sin(lat0)  # Z coordinate
        zr0 = radius * math.cos(lat0)  # Radius in x-y plane

        lat1 = math.pi * (-0.5 + float(i + 1) / slices)  # Latitude 1
        z1 = radius * math.sin(lat1)
        zr1 = radius * math.cos(lat1)

        glBegin(GL_QUAD_STRIP)
        for j in range(stacks + 1):
            lon = 2 * math.pi * float(j) / stacks  # Longitude
            x = zr0 * math.cos(lon)  # X coordinate
            y = zr0 * math.sin(lon)  # Y coordinate
            glVertex3f(x, y, z0)  # Vertex at (x, y, z0)
            x = zr1 * math.cos(lon)  # X coordinate
            y = zr1 * math.sin(lon)  # Y coordinate
            print("----------------------------", x, y, z1)
            glVertex3f(x, y, z1)  # Vertex at (x, y, z1)
        glEnd()


# # Drawing the particle
# def draw_particle(position, radius):
#     glPushMatrix()
#     glTranslatef(position[0], position[1], position[2])
#     draw_sphere(radius)  # Render sphere manually
#     glPopMatrix()

# Initialize Pygame and OpenGL
pygame.init()
display = (1000, 1000)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

# Set up camera (perspective)
gluPerspective(45, (display[0] / display[1]), 0.0, 5000.0)
glTranslatef(0.0, 0.0, 10)  # Move camera back to see the particles

# Simulation parameters
num_objects = 10
positions = wp.array(
    [
        [np.random.uniform(-10, 10), np.random.uniform(-10, 10), 0]
        for _ in range(num_objects)
    ],
    dtype=wp.vec3,
)
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
radius = 1.0  # Adjust radius of the 3D particles
num_frames = 1000  # Number of frames to simulate
fps = 30  # Frames per second for the video
output_filename = "simulation_3d_output.avi"
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(output_filename, fourcc, fps, display)


# OpenGL drawing function for particles
def draw_particle(position):
    glPushMatrix()
    glTranslatef(position[0], position[1], position[2])
    draw_sphere(radius, 10, 10)  # Render sphere with a radius
    glPopMatrix()


# Simulation and rendering loop
for frame_idx in range(num_frames):
    # Simulate one step
    wp.launch(update_positions, dim=num_objects, inputs=[positions, velocities, 0.1])
    wp.launch(
        detect_collisions,
        dim=num_objects,
        inputs=[positions, velocities, radius, num_objects],
    )

    # Clear the screen and set background color (black)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Draw particles in 3D space
    positions_host = positions.numpy()
    for pos in positions_host:
        print(pos)
        draw_particle(pos)

    # Update the screen and capture frame
    pygame.display.flip()
    pygame.time.wait(1000 // fps)  # Wait to maintain the desired FPS

    # Capture the frame for the video
    # frame = np.array(pygame.image.tostring(pygame.display.get_surface(), "RGB"))
    frame = np.array(pygame.surfarray.pixels3d(pygame.display.get_surface()))
    frame = np.flip(frame, axis=0)  # Flip to match OpenGL coordinate system

    frame = frame.reshape((display[1], display[0], 3))
    out.write(frame)

# Release the video writer
out.release()
pygame.quit()
print(f"3D simulation video saved as {output_filename}")
