from enum import IntEnum
import numpy as np
import math
from thegame import Ability

from gym.envs.registration import register
from gym.spaces import Box

from thegame.experimental.gymbase import SinglePlayerEnv, pb2, GameState


class Attr(IntEnum):
    pos_x = 0
    pos_y = 1
    vel_x = 2
    vel_y = 3
    health = 4
    body_damage = 5
    experience = 6
    is_bullet = 7
    is_polygon = 8


skills_to_level = [
    (1, [Ability.HealthRegen]),
    (8, [Ability.Reload]),
    (2, [Ability.MaxHealth, Ability.BulletSpeed]),
    (8, [Ability.BulletDamage, Ability.HealthRegen, Ability.BulletPenetration]),
    (8, range(8))
]


def get_skill_to_level(hero):
    for target_level, skills in skills_to_level:
        skill = min(skills, key=lambda s: hero.abilities[s].level)
        if hero.abilities[skill].level < target_level:
            return skill


def guess_server_path():
    """
    tries to find `thegame` binary.

    It first tires $PATH, then $GOPATH/bin/thegame
    """
    import os
    import shutil
    which = shutil.which('thegame')
    if which is not None:
        return which
    gopaths = os.environ.get('GOPATH', os.path.expanduser('~/go'))
    for gopath in gopaths.split(os.pathsep):
        server_path = os.path.join(gopath, 'bin', 'thegame')
        if shutil.which(server_path) is not None:
            return server_path
    raise Exception("Cannot guess path to thegame's server")


class TheGameEnv(SinglePlayerEnv):
    # Number of steps before thegame is reset.
    # set it here to override the default value (16384).
    # total_steps = 16384

    def __init__(self):
        super().__init__(
            bin=guess_server_path(),
            listen='localhost:50051',  # listen on localhost only
            # listen=':50051',  # listen on all addresses
        )

        self.observation_space = Box(low=-1, high=1, shape=(100, 9))
        self.action_space = Box(low=-1, high=-1, shape=(4,))

    def action_to_controls(self, action):
        level = get_skill_to_level(self.game_state.hero)
        return pb2.Controls(
            level_up=[level],
            accelerate=True,
            acceleration_direction=math.atan2(action[1], action[0]),
            shoot=True,
            shoot_direction=math.atan2(action[3], action[2]),
        )

    def game_state_to_observation(self, gs: GameState):
        obv = np.zeros(
            self.observation_space.shape, self.observation_space.dtype)

        def setentity(i, ent):
            xdiff = ent.position.x - gs.hero.position.x
            ydiff = ent.position.y - gs.hero.position.y
            obv[i, Attr.pos_x] = xdiff / 800
            obv[i, Attr.pos_y] = ydiff / 800
            obv[i, Attr.vel_x] = ent.velocity.x / 40
            obv[i, Attr.vel_y] = ent.velocity.y / 40
            obv[i, Attr.health] = ent.health / 5000
            obv[i, Attr.body_damage] = ent.body_damage / 60
            obv[i, Attr.experience] = ent.rewarding_experience / 360

        for i, polygon in enumerate(gs.polygons):
            setentity(i, polygon)
        for i, bullet in enumerate(gs.bullets):
            setentity(i, bullet)
        return obv

    def get_reward(self, prev, curr):
        return curr.hero.score - prev.hero.score


register(
    id='thegame-v0',
    entry_point='gym_thegame:TheGameEnv',
)
