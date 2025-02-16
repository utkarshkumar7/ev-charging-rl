from env.charging_env import EVChargingEnv

env = EVChargingEnv()
state = env.reset()
print("Initial State:", state)

action = env.action_space.sample()
new_state, reward, done, _ = env.step(action)

print("Action Taken:", action)
print("New State:", new_state)
print("Reward:", reward)
