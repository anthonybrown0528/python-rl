import run_agent
import numpy as np

def test_discrete_space(discretization, dof):
    discrete_action_space = np.linspace([-1.0] * dof, [1.0] * dof, discretization)
    return discrete_action_space

def test_build_action_map(discretization, dof):
    discrete_action_space = np.linspace([-1.0] * dof, [1.0] * dof, discretization)
    action_map = {}

    for idx in range(discretization**dof):
        action_map[idx] = run_agent.get_action_from_discrete(idx, discrete_action_space)
        # print(action_map[idx])
    return action_map


discretization = 2
dof = 4

action_map = test_build_action_map(discretization, dof)
# for key in action_map:
#     print(action_map[key])

# for i in range(16):
#     action_idx = i
#     for joint in range(dof):
#         exp = joint
#         power = discretization**exp

#         action_space_idx = (action_idx // power) % discretization
#         print(action_space_idx)

discrete_action_space = np.linspace([-1.0] * dof, [1.0] * dof, discretization)
for i in range(discretization**dof):
    action_idx = i
    action = np.zeros(4, dtype=np.float32)
    for joint in range(dof):
        exp = joint
        power = discretization**exp

        action_space_idx = (action_idx // power) % discretization
        action[joint] = discrete_action_space[action_space_idx][joint]

    print(action)