# This file is modified from:
# https://github.com/carla-simulator/carla/blob/2c8f55ad3636a8c7999393f0d300fb1434b5fa31/PythonAPI/examples/manual_control.py
# which author: Pierre Sermanet, Kelvin Xu, and Sergey Levine,
# Unsupervised perceptual rewards for imitation learning,
# Proceedings of Robotics: Science and Systems (RSS) (2017).

"""Carla client for RL
    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

try:
    sys.path.append(glob.glob('**/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


import cv2
from cv2 import imread, imwrite, resize, INTER_CUBIC

from keras.models import load_model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import time

from tools_project import get_file


# Used for recording expert demonstrations
class Demonstration():
    # Saves current states of image
    image_throttle = None
    image_steer = None
    image_brake = None
    # Directory count of _outs:
    out_count = 0



# ==============================================================================
# -- Agent ---------- ----------------------------------------------------------
# ==============================================================================

# Contains also reward model
class Agent:
    # References to environment variables:
    _client = None
    _clock = None
    _controller = None
    _display = None
    _pygame = None
    _settings = None
    _synchro_enabled = False
    _world = None

    # Amount to punish when collides or goes outside drive area
    _punish_collision = None
    _reward_goal = None

    # Agent static variables:
    _collision = None
    _out_of_area = None
    _done = False
    _first_image = True
    _image = None
    _prediction = None
    _reward_observation = 0
    _update_needed = True
    _user_control = 0
    _reward_model = None
    _disable_brake = None
    _keep_throttle = -1


    def __init__(self, args, reward_model_path):
        # with tf.device('/device:GPU:0'):
        Agent._reward_model = load_model(reward_model_path)
        Agent._user_control = args.user_control
        Agent._disable_brake = args.disable_brake

    def enable_sync(self):
        if not self._synchro_enabled:
            print('Synchronous mode not enabled in client side.')
            return
            print('Enabling synchronous mode.')
            self._settings = self._world.world.get_settings()
            self._settings.synchronous_mode = True
            self._world.world.apply_settings(self._settings)
            print('Synchronous mode enabled')
            self._synchro_enabled = True

    def env_step(self, action):
        # Reset reward
        Agent._reward_observation = 0
        # If episode is done, wait
        if Agent._done:
            print('Agent: Episode ended, waiting for a new episode start.')
            while Agent._done:
                pass

        # Check that observation exists to get the reward
        if Agent._image is None:
            print('Agent: Waiting for image to get reward')
            while Agent._image is None:
                pass

        self._clock.tick() # Put here fps cap if needed

        # Check key presses etc.
        if self._controller.parse_events(self._client, self._world, self._clock):
            print('Agent: Quit was pressed')
            self.destroy_game()
            return 0, True
        # Control the car with predicted action
        if action is not None and Agent._user_control == 0:
            self._controller._parse_agent_vehicle_keys(action, self._clock.get_time())

        # Update world
        self._world.world.wait_for_tick()
        self._world.tick(self._clock)
        self._world.world.tick()
        # Update screen
        self._world.render(self._display)
        self._pygame.display.flip()

        # Show that a new update is needed
        Agent._update_needed = True

        # Determine is goal reached:

        # Get player location
        t = self._world.player.get_transform()
        # Change done to True if the car reached to goal (under specified area)
        if 150.5 - t.location.x  > 0 and 150.5 - t.location.x < 5.5 and -90 - t.location.y < 0 and -90 - t.location.y > -5:
            # Add extra reward if it brakes here:
            # if action[3] > 0.5:
            #     Agent._reward_observation = 10
            Agent._done = True
            Agent._reward_observation += Agent._reward_goal
        # End episode if drives too far.
        elif t.location.x > 160 or t.location.y > -84 or t.location.y < -132 or t.location.x < 97 or (t.location.y > -126 and t.location.x < 118):
            Agent._done = True
            Agent._out_of_area = True
            Agent._reward_observation += Agent._punish_collision
        # Get the reward
        Agent._prediction = Agent._reward_model.predict(np.expand_dims(Agent._image,axis=0))[0]
        # Add reward
        Agent._reward_observation += Agent._prediction[0]#.argmax() # + 1 - Agent._prediction[0][Agent._prediction.argmax()]

        return Agent._reward_observation

    def env_restart(self):
        # Restart environment
        self._world.restart()
        # Set done to false
        Agent._done = False
        Agent._out_of_area = False
        Agent._collision = False
        # Request update
        Agent._update_needed = True

        if Agent._user_control == 1:
            Demonstration.out_count += 1
            print('Now images subfolders count is: ', Demonstration.out_count)


    def env_start(self,args):

        args.width, args.height = [int(x) for x in args.res.split('x')]

        Agent._keep_throttle = args.keep_throttle
        Agent._user_control = args.user_control
        Agent._punish_collision = args.punish_collision
        Agent._reward_goal = args.reward_goal

        if Agent._user_control == 1:
            # Get amount of _out folders:
            # Folders to count
            base_folder = "./images"
            out_folders = os.listdir(base_folder)
            out_folders.sort()
            for out_folder in out_folders:
                Demonstration.out_count
                Demonstration.out_count += 1
            print("images subfolders count: ", Demonstration.out_count)


        try:

            pygame.init()
            self._pygame = pygame
            self._pygame.font.init()

            print('Connecting to CARLA...')
            self._client = carla.Client('127.0.0.1', 2000)
            self._client.set_timeout(5.0)

            self._display = self._pygame.display.set_mode(
                (args.width, args.height),
                self._pygame.HWSURFACE | self._pygame.DOUBLEBUF)

            hud = HUD(args.width, args.height)

            world_base = self._client.get_world()

            self._world = World(world_base, hud, 'vehicle.*', args.new_location)
            self._controller = KeyboardControl(self._world, False)

            self._clock = self._pygame.time.Clock()

            self._world.camera_manager.toggle_camera()

            self.enable_sync()

            Agent._collision = False
            Agent._out_of_area = False

        except KeyboardInterrupt:
            print('\nCancelled by user. Bye!')


    def destroy_game(self):
        self._world.restart()
        if self._synchro_enabled:
            print('\nDisabling synchronous mode.')
            self._settings = self._world.world.get_settings()
            self._settings.synchronous_mode = False
            self._world.world.apply_settings(self._settings)
            print('Disabled synchronous mode.')

        if (self._world and self._world.recording_enabled):
            self._client.stop_recorder()

        if self._world is not None:
            self._world.destroySensors()
            self._world.destroy()

        self._world = None
        self._pygame.quit()



# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate-1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, actor_filter, new_location):
        self.new_location = new_location

        self.world = carla_world
        self.map = self.world.get_map()
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = actor_filter
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

    def set_start_position(self):
        if self.new_location == 0:
            spawn_point = self.player.get_transform()
            spawn_point.location.z = 8.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            spawn_point.rotation.yaw = 0.0
            spawn_point.location.x = 97.3
            spawn_point.location.y = -129.6
            self.player.location = spawn_point
        else:
            spawn_point = self.player.get_transform()
            spawn_point.location.z = 4.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            spawn_point.rotation.yaw = 0.0
            spawn_point.location.x = -57
            spawn_point.location.y = -134
            self.player.location = spawn_point

    def restart(self):
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager._index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager._transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        ##### Get second actor
        blueprint = self.world.get_blueprint_library().filter(self._actor_filter)[3]
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        # Spawn the player.
        if self.player is not None:
            # Default location
            if self.new_location == 0:
                spawn_point = self.player.get_transform()
                spawn_point.location.z = 8.0
                spawn_point.rotation.roll = 0.0
                spawn_point.rotation.pitch = 0.0
                spawn_point.rotation.yaw = 0.0
                spawn_point.location.x = 97.3
                spawn_point.location.y = -129.6
                self.destroy()
                self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            # New location
            else:
                spawn_point = self.player.get_transform()
                spawn_point.location.z = 4.0
                spawn_point.rotation.roll = 0.0
                spawn_point.rotation.pitch = 0.0
                spawn_point.rotation.yaw = 0.0
                spawn_point.location.x = -57
                spawn_point.location.y = -134
                self.destroy()
                self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            # Default locaiton:
            if self.new_location == 0:
                spawn_points = self.world.get_map().get_spawn_points()
                spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
                spawn_point.location.z = 8.0
                spawn_point.rotation.roll = 0.0
                spawn_point.rotation.pitch = 0.0
                spawn_point.rotation.yaw = 0.0
                spawn_point.location.x = 97.3
                spawn_point.location.y = -129.6
                self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            # New location
            else:
                spawn_points = self.world.get_map().get_spawn_points()
                spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
                spawn_point.location.z = 4.0
                spawn_point.rotation.roll = 0.0
                spawn_point.rotation.pitch = 0.0
                spawn_point.rotation.yaw = 0.0
                spawn_point.location.x = -57
                spawn_point.location.y = -134
                self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager._transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)


    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroySensors(self):
            self.camera_manager.sensor.destroy()
            self.camera_manager.sensor = None
            self.camera_manager._index = None

    def destroy(self):
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()

# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            world.player.set_autopilot(self._autopilot_enabled)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    Demonstration.out_count += 1
                    print('Now images subfolders count is: ', out_count)
                    world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    currentIndex = world.camera_manager._index
                    world.destroySensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(currentIndex)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' % ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not (pygame.key.get_mods() & KMOD_CTRL):
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time())
            world.player.apply_control(self._control)

    # Only run, if defined to player drive
    def _parse_vehicle_keys(self, keys = None, milliseconds = 0):
        if Agent._user_control == 1:
            if Agent._keep_throttle > 0:
                self._control.throttle = Agent._keep_throttle
            else:
                self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
            steer_increment = 5e-4 * milliseconds
            if keys[K_LEFT] or keys[K_a]:
                self._steer_cache -= steer_increment
            elif keys[K_RIGHT] or keys[K_d]:
                self._steer_cache += steer_increment
            else:
                self._steer_cache = 0.0
            self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
            self._control.steer = round(self._steer_cache, 1)
            self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
            self._control.hand_brake = keys[K_SPACE]

    # Updates car controls with action.
    def _parse_agent_vehicle_keys(self, action, milliseconds = 0):
        if Agent._keep_throttle > 0:
            self._control.throttle = Agent._keep_throttle
        else:
            self._control.throttle = 1.0 if action[0] > 0.5 else 0.0

        steer_increment = 5e-4 * milliseconds

        # For BCE model:
        if len(action) == 5:
            # Left
            if action[1] > 0.5:
                self._steer_cache -= steer_increment
            # Right
            if action[3] > 0.5:
                self._steer_cache += steer_increment
            # Straight
            if action[2] > 0.5:
                self._steer_cache = 0.0
            # Brake
            self._control.brake = 1.0 if action[4] > 0.5 else 0.0
        # BCE, straight column depreceated:
        elif len(action) == 4 and Agent._disable_brake == 0:
            # Left
            if action[1] > 0.5:
                self._steer_cache -= steer_increment
            # Right
            if action[2] > 0.5:
                self._steer_cache += steer_increment
            # Straight
            if (action[1] < 0.5 and action[2] < 0.5) or (action[1] > 0.5 and action[2] > 0.5):
                self._steer_cache = 0.0
            # Brake
            self._control.brake = 1.0 if action[3] > 0.5 else 0.0
        # BCE, brake disabled:
        elif len(action) == 3 and Agent._disable_brake == 1:
            # Left
            if action[1] > 0.5:
                self._steer_cache -= steer_increment
            # Right
            if action[2] > 0.5:
                self._steer_cache += steer_increment
            # Straight
            if (action[1] < 0.5 and action[2] < 0.5) or (action[1] > 0.5 and action[2] > 0.5):
                self._steer_cache = 0.0

        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        # For MSE model:
        threshold_min = 0.3333
        threshold_max = 0.6666
        if len(action) == 3 and Agent._disable_brake == 0:
            # Left
            if action[1] < threshold_min:
                self._steer_cache -= steer_increment
            # Right
            elif action[1] > threshold_max:
                self._steer_cache += steer_increment
            # Straight
            else:
                self._steer_cache = 0
            # Brake
            self._control.brake = 1.0 if action[2] > 0.5 else 0.0
        # For MSE model, brake disabled:
        elif len(action) == 2 and Agent._disable_brake == 1:
            # # Left
            if action[1] < threshold_min:
                self._steer_cache -= steer_increment
            # Right
            elif action[1] > threshold_max:
                self._steer_cache += steer_increment
            # Straight
            else:
                self._steer_cache = 0
            # Use direct steer angle:
            # if np.abs(action[1]-0.5) < 0.2:
            #     self._control.steer = 0.0
            # else:
            #     self._control.steer = np.float(action[1]-0.5)

    def _parse_walker_keys(self, keys, milliseconds):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = 5.556 if pygame.key.get_mods() & KMOD_SHIFT else 2.778
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        default_font = 'couriernew'
        mono = default_font #if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame_number = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame_number = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame_number - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
            # Update demonstration information
            Demonstration.image_throttle = c.throttle
            Demonstration.image_steer = c.steer
            Demonstration.image_brake = c.brake
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item: # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._history = []
        self._parent = parent_actor
        self._hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self._history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self._hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self._history.append((event.frame_number, intensity))
        if len(self._history) > 4000:
            self._history.pop(0)

        # Signal that episode ended:
        Agent._done = True
        Agent._collision = True
        Agent._reward_observation += Agent._punish_collision


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self._hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_detector')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        text = ['%r' % str(x).split()[-1] for x in set(event.crossed_lane_markings)]
        self._hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._surface = None
        self._parent = parent_actor
        self._hud = hud
        self._recording = False
        self._camera_transforms = [
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            carla.Transform(carla.Location(x=1.6, z=1.7))]
        self._transform_index = 1
        self._sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self._sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '5000')
            item.append(bp)
        self._index = None

    def toggle_camera(self):
        self._transform_index = (self._transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self._transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self._sensors)
        needs_respawn = True if self._index is None \
            else self._sensors[index][0] != self._sensors[self._index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self._surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self._sensors[index][-1],
                self._camera_transforms[self._transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self._hud.notification(self._sensors[index][2])
        self._index = index

    def next_sensor(self):
        self.set_sensor(self._index + 1)

    def toggle_recording(self):
        self._recording = not self._recording
        self._hud.notification('Recording %s' % ('On' if self._recording else 'Off'))

    def render(self, display):
        if self._surface is not None:
            display.blit(self._surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        # Check is update needed:
        if not Agent._update_needed:
            return
        Agent._update_needed = False
        # Stores copy of array
        array = None
        img = None
        if self._sensors[self._index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0]/3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self._hud.dim) / 100.0
            lidar_data += (0.5 * self._hud.dim[0], 0.5 * self._hud.dim[1])
            lidar_data = np.fabs(lidar_data)
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self._hud.dim[0], self._hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self._surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self._sensors[self._index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self._surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

            # Get image in shape 299x299x3
            img = cv2.cvtColor(cv2.resize(array, (299,299), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
            # Update agent
            Agent._image = img

        if self._recording:
            if Agent._user_control == 1:
                # Skip if not sixth
                # if image.frame_number % 6 != 0:
                #     return
                # Set filename
                directory_filename = "images/out" + str(Demonstration.out_count) + "/"
                if os.path.isdir(directory_filename) != True:
                    os.mkdir(directory_filename)
                image_filename = directory_filename + '%08d' % image.frame_number
                # Save resized image
                imwrite(image_filename+".png",img)
                # image.save_to_disk(image_filename)
                # Save driving info
                with open(image_filename+".txt", "w+") as text_file:
                    print(str(Demonstration.image_throttle)+";"+str(Demonstration.image_steer)+";"+str(Demonstration.image_brake), file=text_file)

