import numpy as np
import warp as wp
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image
import cv2
from pathlib import Path
import math
import time


# Define our material properties structure
@wp.struct
class Material:
    """Material properties combining physics and visual characteristics"""
    density: float  # Mass per unit volume (kg/m³)
    restitution: float  # Bounciness (0-1)
    friction: float  # Surface friction (0-1)
    metallic: float  # Visual metallicness (0-1)
    roughness: float  # Surface roughness (0-1)


# Define our physics object structure
@wp.struct
class PhysicsObject:
    """Physical object with position, forces, and material properties"""
    position: wp.vec3  # Current position
    velocity: wp.vec3  # Linear velocity
    force: wp.vec3  # Accumulated forces
    mass: float  # Object mass
    shape_type: int  # Shape identifier (0=sphere, 1=box, 2=cone, 3=cylinder)
    dimensions: wp.vec3  # Shape dimensions
    material_id: int  # Reference to material properties


# Physics kernel for applying forces and motion
@wp.kernel
def apply_physics(objects: wp.array(dtype=PhysicsObject), dt: float, damping: float):
    """Apply physics forces and update object positions"""
    tid = wp.tid()

    # Apply gravity with realistic scaling
    gravity = wp.vec3(0.0, -9.81, 0.0)  # Standard gravity

    # Instead of wp.rand(), we'll use tid and time to create some variation
    # This creates a small pseudo-random force without needing random number generation
    random_force = wp.vec3(
        wp.sin(float(tid) * dt) * 0.1,  # Varies with object ID and time
        0.0,  # No vertical randomness
        wp.cos(float(tid) * dt + 0.5) * 0.1  # Different pattern for Z axis
    )

    # Accumulate forces
    objects[tid].force += objects[tid].mass * gravity + random_force

    # Update velocity with improved damping
    objects[tid].velocity += (objects[tid].force / objects[tid].mass) * dt
    objects[tid].velocity *= (1.0 - damping * dt)

    # Update position using velocity verlet integration
    objects[tid].position += objects[tid].velocity * dt + \
                             0.5 * (objects[tid].force / objects[tid].mass) * dt * dt

    # Reset forces for next frame
    objects[tid].force = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def handle_collisions(objects: wp.array(dtype=PhysicsObject),
                      materials: wp.array(dtype=Material),
                      num_objects: int):
    """Handle collisions with proper conditional statements"""
    tid = wp.tid()

    # Ground collision with improved bounce
    if objects[tid].position[1] < 0.5:
        material = materials[objects[tid].material_id]

        # Apply position correction
        objects[tid].position = wp.vec3(
            objects[tid].position[0],
            0.5,  # Raised ground level
            objects[tid].position[2]
        )

        # Improved bounce physics
        normal_velocity = objects[tid].velocity[1]
        tangent_velocity = wp.vec3(
            objects[tid].velocity[0],
            0.0,
            objects[tid].velocity[2]
        )

        # Enhanced restitution calculation
        restitution_factor = material.restitution * wp.min(1.0, abs(normal_velocity) / 10.0)
        friction_factor = material.friction * (1.0 - restitution_factor)

        objects[tid].velocity = wp.vec3(
            tangent_velocity[0] * (1.0 - friction_factor),
            -normal_velocity * restitution_factor,
            tangent_velocity[2] * (1.0 - friction_factor)
        )

    # Wall collisions with improved bounds
    wall_elasticity = 0.7
    bounds = 4.5

    # X-axis walls - using standard if-else structure
    if abs(objects[tid].position[0]) > bounds:
        # Declare direction variable
        direction = 0.0

        # Set direction using standard if-else
        if objects[tid].position[0] < 0.0:
            direction = 1.0
        else:
            direction = -1.0

        # Apply position correction
        objects[tid].position = wp.vec3(
            bounds * direction,
            objects[tid].position[1],
            objects[tid].position[2]
        )

        # Apply velocity response
        objects[tid].velocity = wp.vec3(
            -objects[tid].velocity[0] * wall_elasticity,
            objects[tid].velocity[1],
            objects[tid].velocity[2]
        )

    # Z-axis walls - similar structure
    if abs(objects[tid].position[2]) > bounds:
        # Declare direction variable
        direction = 0.0

        # Set direction using standard if-else
        if objects[tid].position[2] < 0.0:
            direction = 1.0
        else:
            direction = -1.0

        # Apply position correction
        objects[tid].position = wp.vec3(
            objects[tid].position[0],
            objects[tid].position[1],
            bounds * direction
        )

        # Apply velocity response
        objects[tid].velocity = wp.vec3(
            objects[tid].velocity[0],
            objects[tid].velocity[1],
            -objects[tid].velocity[2] * wall_elasticity
        )

    # Object-object collisions with improved response
    for j in range(num_objects):
        if tid != j:
            dir = objects[j].position - objects[tid].position
            dist = wp.length(dir)

            # Adjusted collision distance based on object sizes
            min_dist = 0.8

            if dist < min_dist:
                mat1 = materials[objects[tid].material_id]
                mat2 = materials[objects[j].material_id]

                # Calculate average material properties
                avg_restitution = (mat1.restitution + mat2.restitution) * 0.5
                avg_friction = (mat1.friction + mat2.friction) * 0.5

                # Normalize direction
                dir = dir / dist

                # Calculate relative velocity
                rel_vel = objects[j].velocity - objects[tid].velocity

                # Decompose velocity
                normal_vel = wp.dot(rel_vel, dir) * dir
                tangent_vel = rel_vel - normal_vel

                # Enhanced collision response
                impact_speed = wp.length(normal_vel)
                restitution_factor = avg_restitution * wp.min(1.0, impact_speed / 5.0)
                friction_factor = avg_friction * (1.0 - restitution_factor)

                # Apply scaled response
                response = (normal_vel * restitution_factor - tangent_vel * friction_factor) * 0.5
                objects[tid].velocity += response

#
#
# # Physics kernel for handling collisions
# @wp.kernel
# def handle_collisions(objects: wp.array(dtype=PhysicsObject),
#                       materials: wp.array(dtype=Material),
#                       num_objects: int):
#     """Handle collisions between objects and with environment"""
#     tid = wp.tid()
#
#     # Ground collision handling
#     if objects[tid].position[1] < 0.5:
#         material = materials[objects[tid].material_id]
#
#         # Prevent ground penetration
#         penetration = 0.5 - objects[tid].position[1]
#         correction = wp.min(penetration * 0.8, 0.1)
#
#         objects[tid].position = wp.vec3(
#             objects[tid].position[0],
#             0.5 + correction,
#             objects[tid].position[2]
#         )
#
#         # Calculate bounce and friction response
#         normal_velocity = objects[tid].velocity[1]
#         tangent_velocity = wp.vec3(
#             objects[tid].velocity[0],
#             0.0,
#             objects[tid].velocity[2]
#         )
#
#         # Apply material properties to response
#         restitution_factor = material.restitution * wp.min(1.0, abs(normal_velocity) / 10.0)
#         friction_factor = material.friction * (1.0 - restitution_factor)
#
#         # Update velocity with friction
#         tangent_damping = wp.exp(-friction_factor * wp.length(tangent_velocity))
#         objects[tid].velocity = wp.vec3(
#             tangent_velocity[0] * tangent_damping,
#             -normal_velocity * restitution_factor,
#             tangent_velocity[2] * tangent_damping
#         )
#
#     # Wall collision handling
#     bounds = 4.5
#     wall_elasticity = 0.8
#
#     # X-walls
#     if abs(objects[tid].position[0]) > bounds:
#         direction = 1.0 if objects[tid].position[0] < 0.0 else -1.0
#         objects[tid].position = wp.vec3(
#             bounds * direction,
#             objects[tid].position[1],
#             objects[tid].position[2]
#         )
#         objects[tid].velocity = wp.vec3(
#             -objects[tid].velocity[0] * wall_elasticity,
#             objects[tid].velocity[1] * 0.95,
#             objects[tid].velocity[2]
#         )
#
#     # Z-walls
#     if abs(objects[tid].position[2]) > bounds:
#         direction = 1.0 if objects[tid].position[2] < 0.0 else -1.0
#         objects[tid].position = wp.vec3(
#             objects[tid].position[0],
#             objects[tid].position[1],
#             bounds * direction
#         )
#         objects[tid].velocity = wp.vec3(
#             objects[tid].velocity[0],
#             objects[tid].velocity[1] * 0.95,
#             -objects[tid].velocity[2] * wall_elasticity
#         )
#
#     # Object-object collisions
#     for j in range(num_objects):
#         if tid != j:
#             dir = objects[j].position - objects[tid].position
#             dist = wp.length(dir)
#
#             # Collision detection
#             min_dist = 0.8
#             if dist < min_dist:
#                 mat1 = materials[objects[tid].material_id]
#                 mat2 = materials[objects[j].material_id]
#
#                 # Average material properties
#                 avg_restitution = (mat1.restitution + mat2.restitution) * 0.5
#                 avg_friction = (mat1.friction + mat2.friction) * 0.5
#
#                 # Normalize direction
#                 dir = dir / dist
#
#                 # Calculate relative velocity
#                 rel_vel = objects[j].velocity - objects[tid].velocity
#
#                 # Decompose velocity
#                 normal_vel = wp.dot(rel_vel, dir) * dir
#                 tangent_vel = rel_vel - normal_vel
#
#                 # Calculate collision response
#                 impact_speed = wp.length(normal_vel)
#                 restitution_factor = avg_restitution * wp.min(1.0, impact_speed / 5.0)
#                 friction_factor = avg_friction * (1.0 - restitution_factor)
#
#                 # Calculate impulse
#                 reduced_mass = (objects[tid].mass * objects[j].mass) / \
#                                (objects[tid].mass + objects[j].mass)
#                 impulse = (1.0 + restitution_factor) * reduced_mass * normal_vel
#
#                 # Apply response
#                 objects[tid].velocity += (impulse - tangent_vel * friction_factor) * \
#                                          (1.0 / objects[tid].mass)


class PhysicsSimulation:
    def __init__(self, width=800, height=600):
        """Initialize the physics simulation system"""
        print("Initializing Physics Simulation...")

        # Window setup
        self.width = width
        self.height = height
        pygame.init()
        pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)

        # OpenGL setup
        glClearColor(0.2, 0.2, 0.2, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)

        # Initialize components
        self.setup_camera()
        self.setup_lighting()
        self.textures = {}
        self.initialize_textures()

        # Create simulation objects
        self.num_objects = 8
        self.damping = 0.1
        self.materials = self.create_materials()
        self.objects = self.create_objects()

        # Video output setup
        self.setup_video_output(width, height)
        print("Initialization complete")

    def setup_camera(self):
        """Set up the camera view"""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.width / self.height, 0.1, 50.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0.0, 8.0, 15.0,  # Camera position
                  0.0, 0.0, 0.0,  # Look at point
                  0.0, 1.0, 0.0)  # Up vector

    def setup_lighting(self):
        """Configure scene lighting"""
        glLightfv(GL_LIGHT0, GL_POSITION, (10.0, 15.0, 10.0, 1.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.4, 0.4, 0.4, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1.0))

    def setup_video_output(self, width, height):
        """Initialize video output settings for recording the simulation

        Parameters:
            width (int): Width of the output video in pixels
            height (int): Height of the output video in pixels
        """
        # Create output directory if it doesn't exist
        output_dir = Path("simulation_output")
        output_dir.mkdir(exist_ok=True)

        # Set up video writer
        self.output_filename = output_dir / "physics_simulation.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(self.output_filename),
            fourcc,
            30,  # Frame rate (FPS)
            (width, height)  # Frame size
        )
        self.frame_count = 0

    def capture_frame(self):
        """Capture and save the current frame to video"""
        # Read the OpenGL buffer
        buffer = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)

        # Convert buffer to numpy array
        image = np.frombuffer(buffer, dtype=np.uint8)
        image = image.reshape((self.height, self.width, 3))

        # Flip image vertically (OpenGL to image coordinates)
        image = np.flipud(image)

        # Convert RGB to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Write frame to video
        self.video_writer.write(image)
        self.frame_count += 1

    def initialize_textures(self):
        """Load or generate textures for objects"""
        texture_info = {
            'sphere': ('metal.jpg', [192, 192, 192]),
            'box': ('wood.jpg', [139, 69, 19]),
            'cone': ('rubber.jpg', [50, 50, 50]),
            'cylinder': ('glass.jpg', [200, 200, 220])
        }

        for shape_type, (filename, default_color) in texture_info.items():
            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)

            try:
                # Try loading texture from file
                img_path = Path(f"textures/{filename}")
                if img_path.exists():
                    img = Image.open(img_path)
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                    img_data = np.array(img.getdata(), dtype=np.uint8)
                    width, height = img.size
                    print(f"Loaded texture file: {filename}")
                else:
                    raise FileNotFoundError(f"Missing texture: {filename}")

            except Exception as e:
                print(f"Creating procedural texture for {shape_type}")
                width = height = 256
                img_data = self.create_procedural_texture(shape_type, default_color)

            # Set texture parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, img_data)

            self.textures[shape_type] = texture_id

    def create_materials(self):
        """Create physics materials with different properties"""
        dtype = np.dtype([
            ('density', np.float32),
            ('restitution', np.float32),
            ('friction', np.float32),
            ('metallic', np.float32),
            ('roughness', np.float32)
        ])

        materials_data = np.array([
            (7800.0, 0.8, 0.3, 0.9, 0.2),  # Steel
            (1100.0, 0.9, 0.8, 0.0, 0.9),  # Rubber
            (700.0, 0.5, 0.6, 0.1, 0.8),  # Wood
            (2500.0, 0.95, 0.1, 1.0, 0.1)  # Glass
        ], dtype=dtype)

        return wp.array(materials_data, dtype=Material)

    def create_objects(self):
        """Create physics objects with initial properties"""
        dtype = np.dtype([
            ('position', np.float32, 3),
            ('velocity', np.float32, 3),
            ('force', np.float32, 3),
            ('mass', np.float32),
            ('shape_type', np.int32),
            ('dimensions', np.float32, 3),
            ('material_id', np.int32)
        ])

        objects_data = []

        # Create objects in circular arrangement
        for i in range(self.num_objects):
            angle = (i / self.num_objects) * 2 * np.pi
            radius = 3.0

            position = np.array([
                radius * np.cos(angle),
                5.0 + (i % 3),  # Staggered heights
                radius * np.sin(angle)
            ], dtype=np.float32)

            # Add initial velocity towards center
            velocity = np.array([
                -position[0] * 0.2,
                0.0,
                -position[2] * 0.2
            ], dtype=np.float32)

            shape_type = i % 4
            material_id = i % len(self.materials)

            # Set dimensions based on shape
            if shape_type == 0:  # Sphere
                dimensions = np.array([0.5, 0.0, 0.0], dtype=np.float32)
            elif shape_type == 1:  # Box
                dimensions = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            elif shape_type == 2:  # Cone
                dimensions = np.array([0.4, 0.8, 0.0], dtype=np.float32)
            else:  # Cylinder
                dimensions = np.array([0.4, 0.8, 0.0], dtype=np.float32)

            # Calculate mass from volume and density
            volume = self.calculate_volume(shape_type, dimensions)
            mass = float(self.materials.numpy()[material_id]['density'] * volume * 0.001)

            force = np.zeros(3, dtype=np.float32)

            object_data = np.array([(
                position,
                velocity,
                force,
                mass,
                shape_type,
                dimensions,
                material_id
            )], dtype=dtype)

            objects_data.append(object_data)

        return wp.array(np.concatenate(objects_data), dtype=PhysicsObject)


    def create_procedural_texture(self, shape_type, base_color):
        """Generate procedural textures with realistic patterns"""
        size = 256
        texture = np.zeros((size, size, 4), dtype=np.uint8)

        if shape_type == 'sphere':  # Metallic texture
            for i in range(size):
                for j in range(size):
                    # Create metallic pattern with highlights
                    val = int(((math.sin(i / 10) + math.sin(j / 10)) / 2 + 1) * 127)
                    r = min(255, base_color[0] + val)
                    g = min(255, base_color[1] + val)
                    b = min(255, base_color[2] + val)
                    texture[i, j] = [r, g, b, 255]

        elif shape_type == 'box':  # Wood grain texture
            for i in range(size):
                for j in range(size):
                    # Create wood grain effect
                    grain = math.sin(i / 20) * 0.5 + math.sin(j / 2) * 0.5
                    val = int((grain + 1) * 64)
                    r = min(255, base_color[0] + val)
                    g = min(255, base_color[1] + val)
                    b = min(255, base_color[2] + val)
                    texture[i, j] = [r, g, b, 255]

        elif shape_type == 'cone':  # Rubber texture
            texture = np.ones((size, size, 4), dtype=np.uint8) * [*base_color, 255]
            noise = np.random.randint(0, 20, (size, size, 4), dtype=np.uint8)
            texture = np.clip(texture + noise, 0, 255)

        else:  # Glass texture for cylinder
            texture = np.ones((size, size, 4), dtype=np.uint8) * [*base_color, 200]
            noise = np.random.randint(0, 15, (size, size, 4), dtype=np.uint8)
            texture = np.clip(texture + noise, 0, 255)

        return texture.flatten()

    def calculate_volume(self, shape_type, dimensions):
        """Calculate volume for mass computation"""
        if shape_type == 0:  # Sphere
            return (4 / 3) * np.pi * dimensions[0] ** 3
        elif shape_type == 1:  # Box
            return dimensions[0] * dimensions[1] * dimensions[2]
        elif shape_type == 2:  # Cone
            return (1 / 3) * np.pi * dimensions[0] ** 2 * dimensions[1]
        else:  # Cylinder
            return np.pi * dimensions[0] ** 2 * dimensions[1]

    def draw_textured_shape(self, shape_type, dimensions):
        """Draw a shape with its corresponding texture"""
        # Enable texturing
        glEnable(GL_TEXTURE_2D)

        # Bind appropriate texture based on shape
        shape_names = ['sphere', 'box', 'cone', 'cylinder']
        glBindTexture(GL_TEXTURE_2D, self.textures[shape_names[shape_type]])

        if shape_type == 0:  # Sphere
            self.draw_textured_sphere(dimensions[0])
        elif shape_type == 1:  # Box
            self.draw_textured_box(dimensions)
        elif shape_type == 2:  # Cone
            self.draw_textured_cone(dimensions[0], dimensions[1])
        else:  # Cylinder
            self.draw_textured_cylinder(dimensions[0], dimensions[1])

        # Disable texturing after drawing
        glDisable(GL_TEXTURE_2D)

    def draw_textured_sphere(self, radius, slices=32, stacks=32):
        """Draw a textured sphere using quadrics"""
        quad = gluNewQuadric()
        gluQuadricTexture(quad, GL_TRUE)
        gluQuadricNormals(quad, GLU_SMOOTH)
        gluSphere(quad, radius, slices, stacks)
        gluDeleteQuadric(quad)

    def draw_textured_box(self, dimensions):
        """Draw a textured box with proper UV mapping"""
        w, h, d = dimensions * 0.5

        glBegin(GL_QUADS)
        # Front face
        glNormal3f(0, 0, 1)
        glTexCoord2f(0, 0);
        glVertex3f(-w, -h, d)
        glTexCoord2f(1, 0);
        glVertex3f(w, -h, d)
        glTexCoord2f(1, 1);
        glVertex3f(w, h, d)
        glTexCoord2f(0, 1);
        glVertex3f(-w, h, d)

        # Back face
        glNormal3f(0, 0, -1)
        glTexCoord2f(0, 0);
        glVertex3f(w, -h, -d)
        glTexCoord2f(1, 0);
        glVertex3f(-w, -h, -d)
        glTexCoord2f(1, 1);
        glVertex3f(-w, h, -d)
        glTexCoord2f(0, 1);
        glVertex3f(w, h, -d)

        # Top face
        glNormal3f(0, 1, 0)
        glTexCoord2f(0, 0);
        glVertex3f(-w, h, -d)
        glTexCoord2f(1, 0);
        glVertex3f(-w, h, d)
        glTexCoord2f(1, 1);
        glVertex3f(w, h, d)
        glTexCoord2f(0, 1);
        glVertex3f(w, h, -d)

        # Bottom face
        glNormal3f(0, -1, 0)
        glTexCoord2f(0, 0);
        glVertex3f(-w, -h, -d)
        glTexCoord2f(1, 0);
        glVertex3f(w, -h, -d)
        glTexCoord2f(1, 1);
        glVertex3f(w, -h, d)
        glTexCoord2f(0, 1);
        glVertex3f(-w, -h, d)

        # Right face
        glNormal3f(1, 0, 0)
        glTexCoord2f(0, 0);
        glVertex3f(w, -h, -d)
        glTexCoord2f(1, 0);
        glVertex3f(w, h, -d)
        glTexCoord2f(1, 1);
        glVertex3f(w, h, d)
        glTexCoord2f(0, 1);
        glVertex3f(w, -h, d)

        # Left face
        glNormal3f(-1, 0, 0)
        glTexCoord2f(0, 0);
        glVertex3f(-w, -h, -d)
        glTexCoord2f(1, 0);
        glVertex3f(-w, -h, d)
        glTexCoord2f(1, 1);
        glVertex3f(-w, h, d)
        glTexCoord2f(0, 1);
        glVertex3f(-w, h, -d)
        glEnd()

    def draw_textured_cylinder(self, radius, height, slices=32):
        """Draw a textured cylinder with end caps"""
        quad = gluNewQuadric()
        gluQuadricTexture(quad, GL_TRUE)
        gluQuadricNormals(quad, GLU_SMOOTH)

        glPushMatrix()
        glTranslatef(0, -height / 2, 0)

        # Draw cylinder body
        gluCylinder(quad, radius, radius, height, slices, 1)

        # Draw top and bottom caps
        glPushMatrix()
        glRotatef(180, 1, 0, 0)
        gluDisk(quad, 0, radius, slices, 1)
        glPopMatrix()

        glTranslatef(0, 0, height)
        gluDisk(quad, 0, radius, slices, 1)

        glPopMatrix()
        gluDeleteQuadric(quad)

    def draw_textured_cone(self, radius, height, slices=32):
        """Draw a textured cone with bottom cap"""
        quad = gluNewQuadric()
        gluQuadricTexture(quad, GL_TRUE)
        gluQuadricNormals(quad, GLU_SMOOTH)

        glPushMatrix()
        glTranslatef(0, -height / 2, 0)

        # Draw cone body
        gluCylinder(quad, radius, 0, height, slices, 1)

        # Draw bottom cap
        glRotatef(180, 1, 0, 0)
        gluDisk(quad, 0, radius, slices, 1)

        glPopMatrix()
        gluDeleteQuadric(quad)

    def draw_environment(self):
        """Draw the ground and walls"""
        # Ground plane
        glDisable(GL_LIGHTING)
        glBegin(GL_QUADS)
        glColor3f(0.5, 0.5, 0.5)
        glVertex3f(-5.0, 0.0, -5.0)
        glVertex3f(-5.0, 0.0, 5.0)
        glVertex3f(5.0, 0.0, 5.0)
        glVertex3f(5.0, 0.0, -5.0)
        glEnd()

        # Transparent walls
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glBegin(GL_QUADS)
        glColor4f(0.7, 0.7, 0.8, 0.3)

        # Back wall
        glVertex3f(-5.0, 0.0, -5.0)
        glVertex3f(-5.0, 6.0, -5.0)
        glVertex3f(5.0, 6.0, -5.0)
        glVertex3f(5.0, 0.0, -5.0)

        # Side walls
        glVertex3f(-5.0, 0.0, -5.0)
        glVertex3f(-5.0, 6.0, -5.0)
        glVertex3f(-5.0, 6.0, 5.0)
        glVertex3f(-5.0, 0.0, 5.0)

        glVertex3f(5.0, 0.0, -5.0)
        glVertex3f(5.0, 6.0, -5.0)
        glVertex3f(5.0, 6.0, 5.0)
        glVertex3f(5.0, 0.0, 5.0)
        glEnd()

        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)

    def render_scene(self):
        """Render one frame of the simulation"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Draw environment
        self.draw_environment()

        # Draw objects
        objects_data = self.objects.numpy()
        materials_data = self.materials.numpy()

        for obj in objects_data:
            material = materials_data[obj['material_id']]

            glPushMatrix()
            glTranslatef(*obj['position'])

            # Set material properties
            glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE,
                         (1.0, 1.0, 1.0, 1.0))
            glMaterialfv(GL_FRONT, GL_SPECULAR,
                         (material['metallic'], material['metallic'], material['metallic'], 1.0))
            glMaterialf(GL_FRONT, GL_SHININESS,
                        (1.0 - material['roughness']) * 128)

            # Draw the textured shape
            self.draw_textured_shape(obj['shape_type'], obj['dimensions'])

            glPopMatrix()

        pygame.display.flip()

    def simulate_step(self, dt):
        """Perform one step of physics simulation"""
        sub_steps = 4
        sub_dt = dt / sub_steps

        for _ in range(sub_steps):
            wp.launch(
                apply_physics,
                dim=self.num_objects,
                inputs=[self.objects, sub_dt, self.damping]
            )

            wp.launch(
                handle_collisions,
                dim=self.num_objects,
                inputs=[self.objects, self.materials, self.num_objects]
            )

    def run(self, num_frames=1000):
        """Run the main simulation loop"""
        print("Starting simulation...")
        start_time = time.time()

        try:
            for frame in range(num_frames):
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            return

                # Update physics
                self.simulate_step(1.0 / 60.0)

                # Render frame
                self.render_scene()

                # Record frame
                self.capture_frame()

                # Print progress
                if frame % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame / elapsed if elapsed > 0 else 0
                    print(f"Frame {frame}/{num_frames} - FPS: {fps:.1f}")

        except Exception as e:
            print(f"Simulation error: {str(e)}")
            raise

        finally:
            self.video_writer.release()
            pygame.quit()


def main():
    """Main entry point"""
    wp.init()
    device = wp.get_preferred_device()
    print(f"Running simulation on device: {device}")

    try:
        with wp.ScopedDevice(device):
            sim = PhysicsSimulation()
            sim.run()
    except Exception as e:
        print(f"Error running simulation: {e}")
        raise

if __name__ == "__main__":
    main()
