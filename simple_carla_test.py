import carla

host = '127.0.0.1'
port = 2000
client = carla.Client(host, port)
client.set_timeout(2.0)
world = client.get_world()
blueprint = world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
blueprint.set_attribute('role_name', 'hero')
color = '140,0,0'  # Red
blueprint.set_attribute('color', color)
transform = carla.Transform(carla.Location(x=402.242, y=-20.558, z=1.20001),
                            carla.Rotation(pitch=0, yaw=-89.4014, roll=0))
hero_actor = world.spawn_actor(blueprint, transform)
control = carla.VehicleControl()
control.steer = 0
control.throttle = 1
control.brake = 0.0
control.hand_brake = False
control.manual_gear_shift = False


while True:
    hero_actor.apply_control(control)
    print(hero_actor.get_transform())