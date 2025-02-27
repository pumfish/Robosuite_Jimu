import numpy as np
import robosuite as suite

# create environment instance
env = suite.make(
    #env_name="NutAssembly", # try with other tasks like "Stack" and "Door"
    #env_name="PickPlace", # try with other tasks like "Stack" and "Door"
    #env_name="Mstt", # try with other tasks like "Stack" and "Door"
    env_name="Jimu", # try with other tasks like "Stack" and "Door"
    #env_name="Stack", # try with other tasks like "Stack" and "Door"
    robots="UR5e",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    render_gpu_device_id=0,
    control_freq = 20,
)

# reset the environment
env.reset()

for i in range(100000):
    action = np.random.randn(env.robots[0].dof) # sample random action
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display
    if done: env.reset()
    if i % 100 == 0: print(i)