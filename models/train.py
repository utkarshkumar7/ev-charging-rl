from models.q_learning import QLearningAgent
from env.charging_env import EVChargingEnv

env = EVChargingEnv()
agent = QLearningAgent(env)

agent.train(episodes=5000)
print("Training complete!")
