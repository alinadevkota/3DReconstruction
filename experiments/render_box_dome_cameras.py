import os

import cv2
import numpy as np
import warp as wp
import warp.render  # noqa: E501

np.random.seed(0)


class Simulator:
    def __init__(self):
        self.num_cameras = 16  # possible values: 4, 6, 8, 12, 16
        self.camera_radius = 10

        self.fps = 30

        self.outfile_prefix = "multiview"
        self.output_folder = "output/"

        camera_fov = 90.0
        camera_near_plane = 0.1
        camera_far_plane = 100
        self.renderer = wp.render.OpenGLRenderer(
            vsync=False,
            headless=False,
            camera_fov=camera_fov,
            near_plane=camera_near_plane,
            far_plane=camera_far_plane,
        )

        self._init_video_writers()

        view_matrices = self._get_view_matrices()
        instance_ids = [0, 1]
        tile_indices = [instance_ids for _ in range(len(view_matrices))]
        self.renderer.setup_tiled_rendering(tile_indices, view_matrices=view_matrices)
        self.renderer.render_ground()

        self.position = [2.0, 2.0, 2.0]
        self.velocity = [0.01, 0.01, 0.01]
        self.mass = 1

    def _get_view_matrices(self):
        camera_positions = self.generate_dome_positions(
            self.camera_radius, self.num_cameras
        )

        view_matrices = []
        for camera_pos in camera_positions:
            camera_front = np.subtract((0, 0, 0), camera_pos)  # `target_position`
            camera_front = camera_front / np.linalg.norm(camera_front)  # Normalize
            camera_up = (0, 1, 0)

            self.renderer.update_view_matrix(
                cam_pos=camera_pos, cam_front=camera_front, cam_up=camera_up
            )
            view_matrices.append(self.renderer._view_matrix)

        return view_matrices

    @staticmethod
    def generate_dome_positions(radius, num_points):
        positions = []
        for _ in range(num_points):
            # Random angles for uniform distribution
            theta = np.random.uniform(0, 2 * np.pi)  # Azimuthal angle
            phi = np.random.uniform(0, np.pi / 2)  # Polar angle (upper hemisphere)

            # Convert spherical to Cartesian coordinates
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)

            positions.append((x, z, y))
        return positions

    def _init_video_writers(self):
        fourcc = cv2.VideoWriter_fourcc(*"XVID")

        outfiles = [
            os.path.join(self.output_folder, f"{self.outfile_prefix}_{i}.avi")
            for i in range(self.num_cameras)
        ]
        self.video_writers = [
            cv2.VideoWriter(
                outfile,
                fourcc,
                self.fps,
                (
                    int(self.renderer.screen_width / 4),
                    int(self.renderer.screen_height / 4),
                ),
            )
            for outfile in outfiles
        ]

    def render(self):
        time = self.renderer.clock_time
        self.renderer.begin_frame(time)

        self.renderer.render_cylinder(
            "cylinder",
            [3.2, 1.0, np.sin(time + 0.5)],
            np.array(
                wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.sin(time + 0.5))
            ),
            radius=2.5,
            half_height=2.8,
        )

        self.renderer.end_frame()

    def save_frame(self):
        pixel_shape = (
            self.num_cameras,
            self.renderer.screen_height / 4,
            self.renderer.screen_width / 4,
            3,
        )
        pixels = wp.zeros(pixel_shape, dtype=wp.float32)
        self.renderer.get_pixels(pixels, mode="rgb", use_uint8=False)

        for i, writer in enumerate(self.video_writers):
            frame = pixels.numpy()[i]
            frame = np.clip(frame, 0.0, 1.0)
            frame = (frame * 255).astype(np.uint8)
            frame = np.flip(frame, axis=0)
            frame_to_write = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # cv2.imwrite("output/frame.png", frame_to_write)
            writer.write(frame_to_write)


if __name__ == "__main__":
    DEVICE = None

    with wp.ScopedDevice(DEVICE):
        simulator = Simulator()

        for i in range(200):
            simulator.render()
            simulator.save_frame()

        simulator.renderer.clear()
        for video_writer in simulator.video_writers:
            video_writer.release()
