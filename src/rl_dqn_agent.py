"""
Deep Q-Network (DQN) Agent for Reinforcement Learning Trading

Implements a DQN agent with experience replay and target network
for learning optimal trading strategies.
"""

import numpy as np
import pandas as pd
import random
from collections import deque, namedtuple
import pickle
import logging
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import deep learning libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Experience tuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Add experience to buffer."""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch of experiences."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)


if TORCH_AVAILABLE:
    class DQNNetwork(nn.Module):
        """Deep Q-Network using PyTorch."""
        
        def __init__(self, input_size: int, hidden_size: int = 256, output_size: int = 3):
            super(DQNNetwork, self).__init__()
            
            self.network = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, output_size)
            )
        
        def forward(self, x):
            return self.network(x)

elif TF_AVAILABLE:
    def create_dqn_network(input_size: int, hidden_size: int = 256, output_size: int = 3):
        """Create DQN network using TensorFlow/Keras."""
        model = Sequential([
            Dense(hidden_size, input_dim=input_size, activation='relu'),
            Dropout(0.2),
            Dense(hidden_size, activation='relu'),
            Dropout(0.2),
            Dense(hidden_size // 2, activation='relu'),
            Dense(output_size, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse'
        )
        
        return model


class DQNAgent:
    """Deep Q-Network Agent for trading."""
    
    def __init__(
        self,
        state_size: int,
        action_size: int = 3,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 32,
        buffer_size: int = 10000,
        target_update_freq: int = 100
    ):
        """
        Initialize DQN Agent.
        
        Args:
            state_size: Size of state space
            action_size: Size of action space (3: Hold, Buy, Sell)
            learning_rate: Learning rate for neural network
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            batch_size: Batch size for training
            buffer_size: Size of experience replay buffer
            target_update_freq: Frequency to update target network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Experience replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Training tracking
        self.training_step = 0
        self.losses = []
        
        # Initialize networks
        self._initialize_networks()
        
        logger.info(f"DQN Agent initialized")
        logger.info(f"State size: {state_size}, Action size: {action_size}")
        logger.info(f"Using {'PyTorch' if TORCH_AVAILABLE else 'TensorFlow' if TF_AVAILABLE else 'No deep learning'} backend")
    
    def _initialize_networks(self):
        """Initialize main and target networks."""
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Main network
            self.q_network = DQNNetwork(self.state_size, output_size=self.action_size).to(self.device)
            self.target_network = DQNNetwork(self.state_size, output_size=self.action_size).to(self.device)
            
            # Optimizer
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
            
            # Copy weights to target network
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        elif TF_AVAILABLE:
            # Main network
            self.q_network = create_dqn_network(self.state_size, output_size=self.action_size)
            self.target_network = create_dqn_network(self.state_size, output_size=self.action_size)
            
            # Copy weights to target network
            self.target_network.set_weights(self.q_network.get_weights())
            
        else:
            # Fallback to simple linear model
            logger.warning("No deep learning libraries available. Using simple linear model.")
            self.weights = np.random.randn(self.state_size, self.action_size) * 0.1
            self.target_weights = self.weights.copy()
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy."""
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Get Q-values for current state
        q_values = self.predict(state)
        return np.argmax(q_values)
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict Q-values for given state."""
        state = np.array(state).reshape(1, -1)
        
        if TORCH_AVAILABLE:
            self.q_network.eval()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.cpu().numpy().flatten()
                
        elif TF_AVAILABLE:
            return self.q_network.predict(state, verbose=0).flatten()
            
        else:
            # Linear model fallback
            return np.dot(state, self.weights).flatten()
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self) -> Optional[float]:
        """Train the agent on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch of experiences
        batch = self.memory.sample(self.batch_size)
        
        # Prepare batch data
        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch])
        
        if TORCH_AVAILABLE:
            return self._replay_pytorch(states, actions, rewards, next_states, dones)
        elif TF_AVAILABLE:
            return self._replay_tensorflow(states, actions, rewards, next_states, dones)
        else:
            return self._replay_linear(states, actions, rewards, next_states, dones)
    
    def _replay_pytorch(self, states, actions, rewards, next_states, dones):
        """Training step using PyTorch."""
        self.q_network.train()
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def _replay_tensorflow(self, states, actions, rewards, next_states, dones):
        """Training step using TensorFlow."""
        # Predict Q-values for current states
        current_q_values = self.q_network.predict(states, verbose=0)
        
        # Predict Q-values for next states using target network
        next_q_values = self.target_network.predict(next_states, verbose=0)
        
        # Update Q-values
        targets = current_q_values.copy()
        
        for i in range(len(states)):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train the network
        history = self.q_network.fit(states, targets, epochs=1, verbose=0)
        return history.history['loss'][0]
    
    def _replay_linear(self, states, actions, rewards, next_states, dones):
        """Training step using simple linear model."""
        # Simple gradient descent update
        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            reward = rewards[i]
            next_state = next_states[i]
            done = dones[i]
            
            # Current Q-value
            current_q = np.dot(state, self.weights[:, action])
            
            # Target Q-value
            if done:
                target_q = reward
            else:
                next_q_values = np.dot(next_state, self.target_weights)
                target_q = reward + self.gamma * np.max(next_q_values)
            
            # Update weights
            error = target_q - current_q
            self.weights[:, action] += self.learning_rate * error * state
        
        return abs(error) if 'error' in locals() else 0.0
    
    def update_target_network(self):
        """Update target network weights."""
        if TORCH_AVAILABLE:
            self.target_network.load_state_dict(self.q_network.state_dict())
        elif TF_AVAILABLE:
            self.target_network.set_weights(self.q_network.get_weights())
        else:
            self.target_weights = self.weights.copy()
    
    def train_step(self):
        """Perform one training step."""
        loss = self.replay()
        
        if loss is not None:
            self.losses.append(loss)
            self.training_step += 1
            
            # Update target network periodically
            if self.training_step % self.target_update_freq == 0:
                self.update_target_network()
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        
        return loss
    
    def save(self, filepath: str):
        """Save the trained model."""
        model_data = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }
        
        if TORCH_AVAILABLE:
            model_data['model_state_dict'] = self.q_network.state_dict()
            model_data['optimizer_state_dict'] = self.optimizer.state_dict()
        elif TF_AVAILABLE:
            # Save TensorFlow model separately
            self.q_network.save(filepath.replace('.pkl', '_tf_model'))
        else:
            model_data['weights'] = self.weights
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"DQN model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore parameters
        self.epsilon = model_data.get('epsilon', self.epsilon_min)
        self.training_step = model_data.get('training_step', 0)
        
        if TORCH_AVAILABLE and 'model_state_dict' in model_data:
            self.q_network.load_state_dict(model_data['model_state_dict'])
            self.target_network.load_state_dict(model_data['model_state_dict'])
            if 'optimizer_state_dict' in model_data:
                self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
        elif TF_AVAILABLE:
            # Load TensorFlow model
            try:
                self.q_network = tf.keras.models.load_model(filepath.replace('.pkl', '_tf_model'))
                self.target_network = tf.keras.models.load_model(filepath.replace('.pkl', '_tf_model'))
            except:
                logger.warning("Could not load TensorFlow model")
        else:
            if 'weights' in model_data:
                self.weights = model_data['weights']
                self.target_weights = self.weights.copy()
        
        logger.info(f"DQN model loaded from {filepath}")
    
    def get_training_stats(self) -> dict:
        """Get training statistics."""
        return {
            'training_steps': self.training_step,
            'epsilon': self.epsilon,
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'memory_size': len(self.memory)
        }


def test_dqn_agent():
    """Test the DQN agent with dummy data."""
    print("Testing DQN Agent...")
    
    # Create dummy environment
    state_size = 50
    agent = DQNAgent(state_size)
    
    # Test prediction
    dummy_state = np.random.randn(state_size)
    action = agent.act(dummy_state)
    print(f"Sample action: {action}")
    
    # Test experience storage and training
    for i in range(100):
        state = np.random.randn(state_size)
        action = agent.act(state)
        reward = np.random.randn()
        next_state = np.random.randn(state_size)
        done = i == 99
        
        agent.remember(state, action, reward, next_state, done)
        
        if i > 32:  # Start training after enough experiences
            loss = agent.train_step()
            if loss and i % 20 == 0:
                print(f"Step {i}, Loss: {loss:.4f}, Epsilon: {agent.epsilon:.3f}")
    
    # Test save/load
    agent.save("test_dqn_model.pkl")
    
    # Create new agent and load
    new_agent = DQNAgent(state_size)
    try:
        new_agent.load("test_dqn_model.pkl")
        print("Save/load test successful")
    except:
        print("Save/load test failed (expected if no deep learning libraries)")
    
    print("DQN Agent test completed!")


if __name__ == "__main__":
    test_dqn_agent()