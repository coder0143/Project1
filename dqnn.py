import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import copy
import time
from sklearn.metrics import accuracy_score, r2_score
from TorchTrack import TorchTrack
import json

class DQNNetwork(nn.Module):
    """Deep Q-Network for architecture generation"""
    
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class ArchitectureState:
    """Represents the current state of neural network architecture"""
    
    def __init__(self, task_type='classification', max_layers=10):
        self.task_type = task_type
        self.max_layers = max_layers
        self.layers = [64]  # Start with one hidden layer
        self.activations = ['relu']  # Activation functions
        self.learning_rate = 0.01
        self.batch_size = 32
        self.epochs = 100
        self.dropout_rate = 0.0
        
    def to_vector(self, input_size=30):
        """Convert architecture state to vector representation"""
        # Pad layers to max_layers
        padded_layers = self.layers + [0] * (self.max_layers - len(self.layers))
        
        # Encode activations (0: relu, 1: tanh, 2: sigmoid)
        activation_map = {'relu': 0, 'tanh': 1, 'sigmoid': 2}
        padded_activations = [activation_map.get(act, 0) for act in self.activations] + \
                           [0] * (self.max_layers - len(self.activations))
        
        # Normalize hyperparameters
        normalized_lr = np.log10(self.learning_rate) / 3.0 + 1.0  # Log scale normalization
        normalized_bs = self.batch_size / 128.0
        normalized_epochs = self.epochs / 200.0
        
        state_vector = (
            [len(self.layers) / self.max_layers] +  # Number of layers (normalized)
            [layer / 512.0 for layer in padded_layers] +  # Layer sizes (normalized)
            [act / 2.0 for act in padded_activations] +  # Activations (normalized)
            [normalized_lr, normalized_bs, normalized_epochs, self.dropout_rate]
        )
        
        return np.array(state_vector, dtype=np.float32)
    
    def copy(self):
        """Create a deep copy of the current state"""
        new_state = ArchitectureState(self.task_type, self.max_layers)
        new_state.layers = self.layers.copy()
        new_state.activations = self.activations.copy()
        new_state.learning_rate = self.learning_rate
        new_state.batch_size = self.batch_size
        new_state.epochs = self.epochs
        new_state.dropout_rate = self.dropout_rate
        return new_state

class ArchitectureGenerator:
    """Generates and modifies neural network architectures"""
    
    def __init__(self, input_size, output_size, task_type='classification'):
        self.input_size = input_size
        self.output_size = output_size
        self.task_type = task_type
        
    def create_model(self, state):
        """Create PyTorch model from architecture state"""
        layers = []
        prev_size = self.input_size
        
        # Add hidden layers
        for i, size in enumerate(state.layers):
            layers.append(nn.Linear(prev_size, size))
            
            # Add activation
            if i < len(state.activations):
                if state.activations[i] == 'relu':
                    layers.append(nn.ReLU())
                elif state.activations[i] == 'tanh':
                    layers.append(nn.Tanh())
                elif state.activations[i] == 'sigmoid':
                    layers.append(nn.Sigmoid())
            
            # Add dropout if specified
            if state.dropout_rate > 0:
                layers.append(nn.Dropout(state.dropout_rate))
                
            prev_size = size
        
        # Output layer
        layers.append(nn.Linear(prev_size, self.output_size))
        
        # Final activation for task type - NO ACTIVATION FOR MULTICLASS
        # We'll use CrossEntropyLoss which includes softmax
        if self.task_type == 'binary_classification':
            layers.append(nn.Sigmoid())
        # For multiclass and regression, no final activation
        
        return nn.Sequential(*layers)

class DQNNAgent:
    """Deep Q-Network Agent for Neural Architecture Search"""
    
    def __init__(self, state_size, action_size, input_size, output_size, task_type='classification'):
        self.state_size = state_size
        self.action_size = action_size
        self.task_type = task_type
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.batch_size = 32
        self.target_update_freq = 100
        self.steps = 0
        
        # Neural networks
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Architecture components
        self.arch_generator = ArchitectureGenerator(input_size, output_size, task_type)
        
        # Update target network
        self.update_target_network()
        
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
        
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([e[0] for e in batch])).to(self.device)
        actions = torch.LongTensor(np.array([e[1] for e in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([e[2] for e in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in batch])).to(self.device)
        dones = torch.BoolTensor(np.array([e[4] for e in batch])).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.update_target_network()

class ArchitectureActions:
    """Define possible actions for modifying architectures"""
    
    @staticmethod
    def apply_action(state, action_id):
        """Apply action to architecture state"""
        new_state = state.copy()
        
        if action_id == 0:  # Add layer
            if len(new_state.layers) < new_state.max_layers:
                new_layer_size = random.choice([16, 32, 64, 128, 256, 512])
                new_state.layers.append(new_layer_size)
                new_state.activations.append(random.choice(['relu', 'tanh', 'sigmoid']))
                
        elif action_id == 1:  # Remove layer
            if len(new_state.layers) > 1:
                new_state.layers.pop()
                new_state.activations.pop()
                
        elif action_id == 2:  # Modify layer size
            if new_state.layers:
                layer_idx = random.randint(0, len(new_state.layers) - 1)
                new_state.layers[layer_idx] = random.choice([16, 32, 64, 128, 256, 512])
                
        elif action_id == 3:  # Change activation
            if new_state.activations:
                act_idx = random.randint(0, len(new_state.activations) - 1)
                new_state.activations[act_idx] = random.choice(['relu', 'tanh', 'sigmoid'])
                
        elif action_id == 4:  # Adjust learning rate
            new_state.learning_rate *= random.choice([0.5, 0.8, 1.2, 2.0])
            new_state.learning_rate = max(0.0001, min(0.1, new_state.learning_rate))
            
        elif action_id == 5:  # Adjust batch size
            new_state.batch_size = random.choice([16, 32, 64, 128])
            
        elif action_id == 6:  # Adjust epochs
            new_state.epochs = random.choice([50, 100, 150, 200, 250])
            
        elif action_id == 7:  # Adjust dropout
            new_state.dropout_rate = random.choice([0.0, 0.1, 0.2, 0.3, 0.5])
            
        return new_state

class ModelTrainer:
    """Train and evaluate neural network architectures"""
    
    def __init__(self, X_train, y_train, X_test, y_test, task_type='classification', num_classes=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.task_type = task_type
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train_and_evaluate(self, state, experiment_name):
        """Train model with given architecture and return performance metrics"""
        # Initialize TorchTrack
        tracker = TorchTrack(experiment_name=experiment_name)
        
        # Determine output size based on task type
        if self.task_type == 'regression':
            output_size = self.y_train.shape[1] if len(self.y_train.shape) > 1 else 1
        elif self.task_type == 'binary_classification':
            output_size = 1
        else:  # multiclass_classification
            output_size = self.num_classes
        
        # Create model
        arch_gen = ArchitectureGenerator(
            self.X_train.shape[1], 
            output_size,
            self.task_type
        )
        model = arch_gen.create_model(state).to(self.device)
        
        # Define loss and optimizer
        if self.task_type == 'binary_classification':
            criterion = nn.BCELoss()
        elif self.task_type == 'multiclass_classification':
            criterion = nn.CrossEntropyLoss()
        else:  # regression
            criterion = nn.MSELoss()
            
        optimizer = optim.Adam(model.parameters(), lr=state.learning_rate)
        
        # Move data to device
        X_train_tensor = self.X_train.to(self.device)
        y_train_tensor = self.y_train.to(self.device)
        X_test_tensor = self.X_test.to(self.device)
        y_test_tensor = self.y_test.to(self.device)
        
        # Training loop
        start_time = time.time()
        best_performance = float('-inf') if self.task_type == 'regression' else 0.0
        
        for epoch in range(state.epochs):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_train_tensor)
            
            # Calculate loss based on task type
            if self.task_type == 'multiclass_classification':
                loss = criterion(outputs, y_train_tensor.long().squeeze())
            else:
                loss = criterion(outputs, y_train_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                
                if self.task_type == 'binary_classification':
                    predicted = (test_outputs > 0.5).float()
                    performance = accuracy_score(
                        y_test_tensor.cpu().numpy(),
                        predicted.cpu().numpy()
                    )
                    best_performance = max(best_performance, performance)
                elif self.task_type == 'multiclass_classification':
                    predicted = torch.argmax(test_outputs, dim=1)
                    performance = accuracy_score(
                        y_test_tensor.cpu().numpy().flatten(),
                        predicted.cpu().numpy()
                    )
                    best_performance = max(best_performance, performance)
                else:  # regression
                    performance = r2_score(
                        y_test_tensor.cpu().numpy(),
                        test_outputs.cpu().numpy()
                    )
                    best_performance = max(best_performance, performance)
            
            # Log epoch data
            tracker.log_epoch(loss=loss.item(), accuracy=performance)
        
        training_time = time.time() - start_time
        
        # Calculate model complexity (number of parameters)
        model_complexity = sum(p.numel() for p in model.parameters())
        
        # Log final results
        hyperparams = {
            'layers': state.layers,
            'activations': state.activations,
            'learning_rate': state.learning_rate,
            'batch_size': state.batch_size,
            'epochs': state.epochs,
            'dropout_rate': state.dropout_rate
        }
        
        metrics = {
            'performance': best_performance,
            'training_time': training_time,
            'model_complexity': model_complexity
        }
        
        if self.task_type in ['binary_classification', 'multiclass_classification']:
            metrics['accuracy'] = best_performance
        else:
            metrics['r2_score'] = best_performance
            
        tracker.log(
            hyperparameters=hyperparams,
            metrics=metrics,
            model_type=self.task_type.capitalize(),
            model_data=f"Generated architecture: {state.layers} layers with {state.activations} activations"
        )
        
        return best_performance, training_time, model_complexity

class RewardCalculator:
    """Calculate rewards for DQNN agent"""
    
    def __init__(self, alpha=1.0, beta=0.001, gamma=0.01):
        self.alpha = alpha  # Performance weight
        self.beta = beta    # Complexity penalty weight
        self.gamma = gamma  # Time penalty weight
        
    def calculate_reward(self, performance, complexity, training_time):
        """Calculate reward based on performance, complexity, and time"""
        # Normalize components
        normalized_performance = performance  # Already between 0-1 for accuracy, can be negative for R²
        normalized_complexity = min(complexity / 1000000, 1.0)  # Normalize by 1M parameters
        normalized_time = min(training_time / 300, 1.0)  # Normalize by 5 minutes
        
        reward = (self.alpha * normalized_performance - 
                 self.beta * normalized_complexity - 
                 self.gamma * normalized_time)
        
        return reward

class DQNN:
    """Main DQNN class for Neural Architecture Search"""
    
    def __init__(self, X_train, y_train, X_test, y_test, task_type='classification'):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        # Determine task type and number of classes
        self.num_classes = None
        if task_type == 'classification':
            # Auto-detect binary vs multiclass
            unique_classes = torch.unique(y_train)
            if len(unique_classes) == 2:
                self.task_type = 'binary_classification'
                # Ensure y is shaped correctly for binary classification
                if len(y_train.shape) == 1:
                    self.y_train = y_train.unsqueeze(1).float()
                    self.y_test = y_test.unsqueeze(1).float()
            else:
                self.task_type = 'multiclass_classification'
                self.num_classes = len(unique_classes)
                # Ensure y is shaped correctly for multiclass
                if len(y_train.shape) > 1 and y_train.shape[1] > 1:
                    # Already one-hot encoded, convert to class indices
                    self.y_train = torch.argmax(y_train, dim=1).unsqueeze(1)
                    self.y_test = torch.argmax(y_test, dim=1).unsqueeze(1)
                elif len(y_train.shape) == 1:
                    self.y_train = y_train.unsqueeze(1)
                    self.y_test = y_test.unsqueeze(1)
        else:
            self.task_type = 'regression'
            # Ensure y is shaped correctly for regression
            if len(y_train.shape) == 1:
                self.y_train = y_train.unsqueeze(1).float()
                self.y_test = y_test.unsqueeze(1).float()
        
        # Initialize components
        self.state_size = 25  # Size of state vector
        self.action_size = 8  # Number of possible actions
        
        # Determine output size for agent
        if self.task_type == 'binary_classification':
            output_size = 1
        elif self.task_type == 'multiclass_classification':
            output_size = self.num_classes
        else:  # regression
            output_size = self.y_train.shape[1] if len(self.y_train.shape) > 1 else 1
        
        self.agent = DQNNAgent(
            self.state_size, 
            self.action_size,
            X_train.shape[1],
            output_size,
            self.task_type
        )
        
        self.trainer = ModelTrainer(
            self.X_train, self.y_train, self.X_test, self.y_test, 
            self.task_type, self.num_classes
        )
        self.reward_calculator = RewardCalculator()
        
        self.best_architecture = None
        self.best_performance = float('-inf') if self.task_type == 'regression' else 0.0
        self.experiment_count = 0
        
    def run_episode(self, max_steps=10):
        """Run one episode of architecture search"""
        # Initialize random architecture
        current_state = ArchitectureState(self.task_type)
        current_state_vector = current_state.to_vector()
        
        total_reward = 0
        
        for step in range(max_steps):
            # Choose action
            action = self.agent.act(current_state_vector)
            
            # Apply action to get new state
            next_state = ArchitectureActions.apply_action(current_state, action)
            next_state_vector = next_state.to_vector()
            
            # Train and evaluate architecture
            self.experiment_count += 1
            experiment_name = f"DQNN_Experiment_{self.experiment_count}"
            
            try:
                performance, training_time, complexity = self.trainer.train_and_evaluate(
                    next_state, experiment_name
                )
                
                # Calculate reward
                reward = self.reward_calculator.calculate_reward(
                    performance, complexity, training_time
                )
                
                # Update best architecture if needed
                if performance > self.best_performance:
                    self.best_performance = performance
                    self.best_architecture = next_state.copy()
                    
                # Store experience
                done = step == max_steps - 1
                self.agent.remember(
                    current_state_vector, action, reward, next_state_vector, done
                )
                
                total_reward += reward
                
                # Train agent
                self.agent.replay()
                
                # Move to next state
                current_state = next_state
                current_state_vector = next_state_vector
                
                print(f"Step {step+1}: Performance={performance:.4f}, Reward={reward:.4f}")
                
            except Exception as e:
                print(f"Error in step {step+1}: {e}")
                # Give negative reward for invalid architectures
                reward = -1.0
                done = True
                self.agent.remember(
                    current_state_vector, action, reward, next_state_vector, done
                )
                break
                
        return total_reward
    
    def train(self, episodes=50, max_steps_per_episode=10):
        """Train the DQNN agent"""
        print(f"Starting DQNN training for {episodes} episodes...")
        print(f"Task type: {self.task_type}")
        if self.num_classes:
            print(f"Number of classes: {self.num_classes}")
        
        # Clear previous TorchTrack data
        tracker = TorchTrack()
        tracker.clean_previous_data()
        
        episode_rewards = []
        
        for episode in range(episodes):
            print(f"\nEpisode {episode+1}/{episodes}")
            total_reward = self.run_episode(max_steps_per_episode)
            episode_rewards.append(total_reward)
            
            print(f"Episode {episode+1} total reward: {total_reward:.4f}")
            print(f"Best performance so far: {self.best_performance:.4f}")
            print(f"Current epsilon: {self.agent.epsilon:.4f}")
            
            # Save best architecture periodically
            if episode % 10 == 0 and self.best_architecture:
                self.save_best_architecture()
                
        print(f"\nTraining completed!")
        print(f"Best performance achieved: {self.best_performance:.4f}")
        
        if self.best_architecture:
            print(f"Best architecture: {self.best_architecture.layers}")
            print(f"Best activations: {self.best_architecture.activations}")
            print(f"Best hyperparameters: LR={self.best_architecture.learning_rate}, "
                  f"BS={self.best_architecture.batch_size}, "
                  f"Epochs={self.best_architecture.epochs}")
            
        return episode_rewards
    
    def save_best_architecture(self):
        """Save the best architecture found"""
        if self.best_architecture:
            best_arch_data = {
                'layers': self.best_architecture.layers,
                'activations': self.best_architecture.activations,
                'learning_rate': self.best_architecture.learning_rate,
                'batch_size': self.best_architecture.batch_size,
                'epochs': self.best_architecture.epochs,
                'dropout_rate': self.best_architecture.dropout_rate,
                'performance': self.best_performance,
                'task_type': self.task_type,
                'num_classes': self.num_classes
            }
            
            with open('best_architecture.json', 'w') as f:
                json.dump(best_arch_data, f, indent=4)
                
            print(f"Best architecture saved to best_architecture.json")
