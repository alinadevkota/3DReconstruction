import numpy as np
import warp as wp
import warp.render
from PIL import Image


class Example:
    def __init__(self):
        self.renderer = wp.render.OpenGLRenderer(vsync=False, headless=False)
        instance_ids = [[0, 1]]
        self.renderer.setup_tiled_rendering(instance_ids)
        self.renderer.render_ground()

        self.pixel_shape = (
            1,
            self.renderer.screen_height,
            self.renderer.screen_width,
            3,
        )

        self.count = 0

        self.position = [2.0, 2.0, 2.0]
        self.velocity = [0.01, 0.01, 0.01]
        self.mass = 1

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
        pixels = wp.zeros(self.pixel_shape, dtype=wp.float32)
        # print(pixels.shape)
        # print(type(pixels))
        render_mode = "rgb"
        example.renderer.get_pixels(pixels, mode=render_mode)
        print(pixels.shape)

        pixels_np = pixels.numpy() * 255
        pixels_np = pixels_np.astype(np.uint8)
        pixels_np = pixels_np[0]

        pil_image = Image.fromarray(pixels_np)
        pil_image.save(f"frames/{self.count:08d}.jpg")
        self.count += 1


if __name__ == "__main__":
    DEVICE = None

    with wp.ScopedDevice(DEVICE):
        example = Example()

        while example.renderer.is_running():
            example.render()
            example.save_frame()

        example.renderer.clear()
