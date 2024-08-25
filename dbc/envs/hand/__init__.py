from gym.envs.registration import register
from dbc.envs.gym_hand import GymHandInterface
from rlf.envs.env_interface import register_env_interface


register(
    id="HandReachCustom-v0",
    entry_point="dbc.envs.hand.reach:HandReachEnv",
    kwargs={"reward_type": "sparse"},
    max_episode_steps=50,
)

register(
        id='CustomHandManipulateBlockRotateZ-v0',
        entry_point='dbc.envs.hand.manipulate:HandBlockEnv',
        kwargs={'target_position': 'ignore', 'target_rotation': 'z', 'reward_type': 'dense'},
        max_episode_steps=50,
    )

register(
        id='CustomHandManipulateBlockRotateZ-v1',
        entry_point='dbc.envs.hand.manipulate_v1:HandBlockEnv',
        kwargs={'target_position': 'ignore', 'target_rotation': 'z', 'reward_type': 'dense'},
        max_episode_steps=50,
    )

register(
        id='CustomHandManipulateBlockRotateZ-v2',
        entry_point='dbc.envs.hand.manipulate_v2:HandBlockEnv',
        kwargs={'target_position': 'ignore', 'target_rotation': 'z', 'reward_type': 'dense'},
        max_episode_steps=50,
    )

register(
        id='CustomHandManipulateBlockRotateZ-v3',
        entry_point='dbc.envs.hand.manipulate_v3:HandBlockEnv',
        kwargs={'target_position': 'ignore', 'target_rotation': 'z', 'reward_type': 'dense'},
        max_episode_steps=50,
    )
register_env_interface("HandReachCustom-v0", GymHandInterface)
register_env_interface("CustomHandManipulateBlockRotateZ-v0", GymHandInterface)
register_env_interface("CustomHandManipulateBlockRotateZ-v1", GymHandInterface)
register_env_interface("CustomHandManipulateBlockRotateZ-v2", GymHandInterface)
register_env_interface("CustomHandManipulateBlockRotateZ-v3", GymHandInterface)
