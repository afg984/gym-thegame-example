from enum import IntEnum
import numpy as np
import math
from thegame import Ability

from gym.envs.registration import register
from gym.spaces import Box

from thegame.experimental.gymbase import SinglePlayerEnv, GameState, Controls


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

    def __init__(self, listen='localhost:50051'):
        import random
        listen = f'localhost:{random.randrange(50000, 60000)}'
        super().__init__(
            server_bin=guess_server_path(),
            listen=listen,
            total_steps=10000,
        )

        state_ranges = [
            (-800, 800),
            (-800, 800),
            (-14, 14),
            (-14, 14),
            (0, 5000),
            (0, 60),
            (0, 360),
            (0, 1),
            (0, 1),
        ]
        low_state = np.array([[x[0] for x in state_ranges]] * 100, dtype=np.float32)
        high_state = np.array([[x[1] for x in state_ranges]] * 100, dtype=np.float32)

        self.observation_space = Box(low=low_state, high=high_state)
        self.action_space = Box(low=-1, high=1, shape=(4,), dtype=np.float32)

    def action_to_controls(self, action):
        level = get_skill_to_level(self.game_state.hero)
        return Controls(
            level_up=[level],
            accelerate=True,
            acceleration_direction=math.atan2(action[1], action[0]),
            shoot=True,
            shoot_direction=math.atan2(action[3], action[2]),
        )

    def game_state_to_observation(self, gs: GameState):
        obv = np.zeros(
            self.observation_space.shape, self.observation_space.dtype)

        def setentity(i, ent, type):
            xdiff = ent.position.x - gs.hero.position.x
            ydiff = ent.position.y - gs.hero.position.y
            obv[i, Attr.pos_x] = xdiff
            obv[i, Attr.pos_y] = ydiff
            obv[i, Attr.vel_x] = ent.velocity.x
            obv[i, Attr.vel_y] = ent.velocity.y
            obv[i, Attr.health] = ent.health
            obv[i, Attr.body_damage] = ent.body_damage
            obv[i, Attr.experience] = ent.rewarding_experience
            obv[i, type] = 1

        setentity(0, gs.hero, 0)
        i = 1
        for i, polygon in enumerate(gs.polygons, 1):
            if i == 100: break
            setentity(i, polygon, Attr.is_polygon)
        for i, bullet in enumerate(gs.bullets, i):
            if i == 100: break
            setentity(i, bullet, Attr.is_bullet)
        return obv

    def get_reward(self, prev, curr):
        return curr.hero.score - prev.hero.score


register(
    id='thegame-v0',
    entry_point='gym_thegame:TheGameEnv',
)
