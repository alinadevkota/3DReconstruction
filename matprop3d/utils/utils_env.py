
from matprop3d.utils.utils_sim import create_body, Box, Sphere


class PoolEnvironment():
    
    def __init__(self, builder, ball_properties):
        self.builder = builder
        self.pg_position_height = 0.1
        self.pg_floor_thickness = 0.1
        self.ball_properties = ball_properties
         
        self.add_balls()
        self.add_playground()
        
    def add_playground(self):
        pg_position_height = self.pg_position_height
        pg_width = 5.0
        pg_length = 5.0
        pg_floor_thickness = self.pg_floor_thickness
        pg_wall_thickness = 0.1
        pg_wall_height = 0.5   
       

        playground_parts = [
            Box(trans=(0,0,0), dim=(pg_width,pg_floor_thickness,pg_length)),
            Box(trans=(pg_width/2,pg_floor_thickness+pg_wall_height/2,0), dim=(pg_wall_thickness,pg_floor_thickness+pg_wall_height,pg_length)), # Right wall
            Box(trans=(-pg_width/2,pg_floor_thickness+pg_wall_height/2,0), dim=(pg_wall_thickness,pg_floor_thickness+pg_wall_height,pg_length)), # Left wall
            Box(trans=(0,pg_floor_thickness+pg_wall_height/2,pg_length/2), dim=(pg_width,pg_floor_thickness+pg_wall_height,pg_wall_thickness)), # Front wall
            Box(trans=(0,pg_floor_thickness+pg_wall_height/2,-pg_length/2), dim=(pg_width,pg_floor_thickness+pg_wall_height,pg_wall_thickness)), # Back wall
        ]

        create_body(self.builder, 'playground', (0,pg_position_height,0), (0,0,0,1), playground_parts)

    def add_balls(self):
        # ball_properties = [
        # #   [Radius, Density, (x,y)]
        #     [0.3, 500, (1.9,-1)],
        #     [0.2, 5000, (1,0)],
        #     # [0.32, 100, (-1,-1)],
        #     # [0.2, 150, (1,0.6)],
        #     # [0.31, 100, (0.8,-0.6)]
        # ]

        for i, ball in enumerate(self.ball_properties):
            ball_radius, ball_density, ball_xy = ball

            ball_parts = [
                Sphere(radius=ball_radius, density=ball_density)
            ]

            create_body(
                self.builder, f"ball_{i}", 
                (ball_xy[0], self.pg_position_height+self.pg_floor_thickness/2+ball_radius, 
                 ball_xy[1]), (0,0,0,1), ball_parts
            )