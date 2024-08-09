import sys
import gym
import numpy as np
import os
import time
import random
from collections import namedtuple
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Define a tensorboard writer for logging
writer = SummaryWriter("./tb_record_halfcheetah")

# Function to perform a soft update of the target network's parameters
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# Function to perform a hard update of the target network's parameters
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

# Define a named tuple to store transitions in the replay memory
Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))

# Replay memory class for storing and sampling experiences
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity  # Maximum size of the replay memory
        self.memory = []  # List to store experiences
        self.position = 0  # Pointer to the current position in the memory

    # Method to add a new experience to the memory
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)  # Add a new slot if memory is not full
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity  # Circular buffer

    # Method to sample a batch of experiences from memory
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    # Return the current size of the memory
    def __len__(self):
        return len(self.memory)

# Ornstein-Uhlenbeck noise for exploration in continuous action spaces
class OUNoise:

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension  # Dimension of the action space
        self.scale = scale  # Scaling factor for the noise
        self.mu = mu  # Mean of the noise
        self.theta = theta  # Theta parameter for OU process
        self.sigma = sigma  # Sigma parameter for OU process
        self.state = np.ones(self.action_dimension) * self.mu  # Initialize noise state
        self.reset()

    # Reset the noise state
    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    # Generate noise
    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale    

# Actor network (policy network)
class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]  # Number of outputs is equal to action dimensions

        # Define the layers of the network
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_outputs)
        
    # Forward pass of the actor network
    def forward(self, inputs):
        x = F.relu(self.linear1(inputs))  # Apply ReLU activation function
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))  # Apply Tanh activation function
        return x

# Critic network (value network)
class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        # Define the layers of the network
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size + num_outputs, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

    # Forward pass of the critic network
    def forward(self, inputs, actions):
        x = F.relu(self.linear1(inputs))
        x = torch.cat((x, actions), 1)  # Concatenate state and action
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

# DDPG agent class
class DDPG(object):
    def __init__(self, num_inputs, action_space, gamma=0.995, tau=0.0005, hidden_size=128, lr_a=1e-4, lr_c=1e-3):

        self.num_inputs = num_inputs  # Number of inputs (state dimensions)
        self.action_space = action_space  # Action space

        # Initialize actor and critic networks along with their target networks
        self.actor = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_target = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_perturbed = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_optim = Adam(self.actor.parameters(), lr=lr_a)  # Optimizer for actor network

        self.critic = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_target = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr_c)  # Optimizer for critic network

        self.gamma = gamma  # Discount factor
        self.tau = tau  # Soft update parameter

        # Copy the weights of the actor and critic to their respective target networks
        hard_update(self.actor_target, self.actor) 
        hard_update(self.critic_target, self.critic)

    # Select an action given the current state
    def select_action(self, state, action_noise=None):
        self.actor.eval()  # Set actor to evaluation mode
        mu = self.actor((Variable(state)))
        mu = mu.data

        # Add noise to the action for exploration
        if action_noise is not None:
            mu += torch.Tensor(action_noise.noise())
        mu = mu.clamp(-1, 1)  # Clamp the action values to the valid range
        return mu

    # Update the parameters of the networks
    def update_parameters(self, batch):
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        mask_batch = Variable(torch.cat(batch.mask))
        next_state_batch = Variable(torch.cat(batch.next_state))
        
        # Compute the target values
        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch)
        
        reward_batch = reward_batch.unsqueeze(1)
        mask_batch = mask_batch.unsqueeze(1)
        expected_state_action_values = reward_batch + (self.gamma * mask_batch * next_state_action_values)

        # Update the critic network
        self.critic_optim.zero_grad()
        state_action_values = self.critic(state_batch, action_batch)
        value_loss = F.mse_loss(state_action_values, expected_state_action_values)
        value_loss.backward()
        self.critic_optim.step()

        # Update the actor network
        self.actor_optim.zero_grad()
        policy_loss = -self.critic(state_batch, self.actor(state_batch))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Soft update the target networks
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    # Save the trained models
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        local_time = time.localtime()
        timestamp = time.strftime("%m%d%Y_%H%M%S", local_time)
        if not os.path.exists('preTrained/'):
            os.makedirs('preTrained/')

        if actor_path is None:
            actor_path = "preTrained/ddpg_actor_{}_{}_{}".format(env_name, timestamp, suffix) 
        if critic_path is None:
            critic_path = "preTrained/ddpg_critic_{}_{}_{}".format(env_name, timestamp, suffix) 
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load the trained models
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None: 
            self.critic.load_state_dict(torch.load(critic_path))

# Training function for the DDPG agent
def train():    
    num_episodes = 300  # Number of training episodes
    gamma = 0.995  # Discount factor
    tau = 0.002  # Soft update parameter
    hidden_size = 128  # Size of hidden layers in the networks
    noise_scale = 0.3  # Scale for the exploration noise
    replay_size = 100000  # Size of the replay memory
    batch_size = 128  # Batch size for updating the networks
    updates_per_step = 1  # Number of updates per step
    print_freq = 1  # Frequency of printing the training progress
    ewma_reward = 0  # Exponentially weighted moving average of rewards
    rewards = []  # List to store episode rewards
    ewma_reward_history = []  # List to store EWMA rewards
    total_numsteps = 0  # Total number of steps taken
    updates = 0  # Total number of updates performed
    
    agent = DDPG(env.observation_space.shape[0], env.action_space, gamma, tau, hidden_size)
    ounoise = OUNoise(env.action_space.shape[0])
    memory = ReplayMemory(replay_size)
    
    for i_episode in range(num_episodes):
        
        ounoise.scale = noise_scale  # Adjust noise scale
        ounoise.reset()  # Reset noise state
        
        state = torch.Tensor([env.reset()])  # Reset environment and get initial state

        episode_reward = 0  # Initialize episode reward
        while True:
            
            action = agent.select_action(state, ounoise)  # Select action with exploration noise
            next_state, reward, done, _ = env.step(action.numpy()[0])
            episode_reward += reward

            mask = torch.Tensor([not done])
            next_state = torch.Tensor([next_state])
            reward = torch.Tensor([reward])

            memory.push(state, action, mask, next_state, reward)  # Store transition in memory

            state = next_state

            if len(memory) > batch_size:
                for _ in range(updates_per_step):
                    transitions = memory.sample(batch_size)  # Sample a batch of experiences
                    batch = Transition(*zip(*transitions))
                    value_loss, policy_loss = agent.update_parameters(batch)  # Update networks

                    writer.add_scalar('loss/value', value_loss, updates)  # Log value loss
                    writer.add_scalar('loss/policy', policy_loss, updates)  # Log policy loss
                    updates += 1

            if done:
                break
           
        rewards.append(episode_reward)  # Store episode reward
        t = 0
        if i_episode % print_freq == 0:
            state = torch.Tensor([env.reset()])
            episode_reward = 0
            while True:
                action = agent.select_action(state)

                next_state, reward, done, _ = env.step(action.numpy()[0])
                
                env.render(mode='human')  # Render the environment
                
                episode_reward += reward

                next_state = torch.Tensor([next_state])

                state = next_state
                
                t += 1
                if done:
                    break

            rewards.append(episode_reward)
            ewma_reward = 0.05 * episode_reward + (1 - 0.05) * ewma_reward  # Update EWMA reward
            ewma_reward_history.append(ewma_reward)           
            print("Episode: {}, length: {}, reward: {:.2f}, ewma reward: {:.2f}".format(i_episode, t, rewards[-1], ewma_reward))
    
    agent.save_model(env_name, '.pth')  # Save the trained models      

# Function for evaluating the trained model
def test(actor_path, critic_path, num_episodes=10):
    agent = DDPG(env.observation_space.shape[0], env.action_space)
    agent.load_model(actor_path, critic_path)  # Load the trained models

    for i_episode in range(num_episodes):
        state = torch.Tensor([env.reset()])
        episode_reward = 0
        
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action.numpy()[0])
            env.render()  # Render the environment
            
            episode_reward += reward
            next_state = torch.Tensor([next_state])
            state = next_state
            
            if done:
                break
        
        print("Episode: {}, reward: {:.2f}".format(i_episode, episode_reward))

if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 10  
    env_name = 'HalfCheetah-v3'
    env = gym.make(env_name)
    env.seed(random_seed)  
    torch.manual_seed(random_seed)  
    train()
    # Uncomment to load the saved checkpoints and evaluate the pre-trained model
    # Specify the paths to your pretrained model files, assure that the paths are correct.

    # actor_path = "preTrained/ddpg_actor_HalfCheetah-v3_05112024_183235_.pth"
    # critic_path = "preTrained/ddpg_critic_HalfCheetah-v3_05112024_183235_.pth"
    # test(actor_path, critic_path)
