# DQNN: Deep Q-Network for Neural Architecture Search

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Deep Reinforcement Learning framework that automatically discovers optimal neural network architectures for your machine learning tasks. DQNN uses Deep Q-Learning to explore the architecture search space and find the best model configuration for classification and regression problems.

## 🌟 Features

- **Automatic Architecture Search**: Discovers optimal neural network architectures using Deep Q-Learning
- **Multi-Task Support**: 
  - Binary Classification
  - Multiclass Classification
  - Regression
- **Smart Hyperparameter Tuning**: Automatically optimizes learning rate, batch size, epochs, and dropout
- **Experiment Tracking**: Built-in tracking system (TorchTrack) for all experiments
- **Interactive Dashboard**: Beautiful Streamlit dashboard for visualization and analysis
- **AI-Powered Insights**: Integration with Ollama for intelligent model recommendations
- **Export & Sharing**: Export results, best architectures, and generated PyTorch code
- **Automatic GPU Dectection**: If using cuda enabled gpu, the code auto detects the gpu and uses it

## 📋 Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Dashboard](#dashboard)
- [AI Insights with Ollama](#ai-insights-with-ollama)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        DQNN Framework                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   DQN Agent  │───▶│ Architecture │───▶│    Model    │   │
│  │  (RL Brain)  │    │  Generator   │    │   Trainer    │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│         │                    │                    │         │
│         ▼                    ▼                    ▼         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Experience Replay Buffer                │   │
│  └──────────────────────────────────────────────────────┘   │
│         │                             ▲             │       │
│         ▼                             |             ▼       │
│  ┌──────────────┐                     |    ┌──────────────┐ │
│  │  TorchTrack  │                     |____│   Reward     │ │
│  │  (Tracking)  │                          │ Calculator   │ │
│  └──────────────┘                          └──────────────┘ │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           Streamlit Dashboard + Ollama AI            │   │
│  └──────────────────────────────────────────────────────┘   │  
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training
- (Optional) Ollama for AI insights

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/dqnn.git
cd dqnn
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### Step 3: Install Python Dependencies

```bash
pip install torch torchvision torchaudio
pip install scikit-learn numpy pandas matplotlib seaborn
pip install streamlit plotly
pip install ollama  # For AI insights
```

Or install all at once:

```bash
pip install -r requirements.txt
```

### Step 4: Install Ollama (Optional, for AI Insights)

#### On macOS/Linux:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### On Windows:
Download and install from [https://ollama.com/download](https://ollama.com/download)

#### Pull a Model:
```bash
ollama pull gemma2:2b
# or
ollama pull llama3.2:3b
# or any other model you prefer
```

**Note**: Update the model name in `app.py` if you use a different model:
```python
md = "gemma2:2b"  # Change this to your preferred model
```

## 🎯 Quick Start

### 1. Run a Simple Example

```python
from dqnn import DQNN
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and prepare data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# Initialize and train DQNN
dqnn = DQNN(X_train, y_train, X_test, y_test, task_type='classification')
episode_rewards = dqnn.train(episodes=20, max_steps_per_episode=8)

print("Training completed! Check the dashboard for results.")
```

### 2. Launch the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📖 Usage

### Using the Provided Examples

Run the pre-configured examples in `model.py`:

```bash
python model.py
```

This will run:
- Binary Classification (Breast Cancer dataset)
- Multiclass Classification (Iris dataset) - commented out by default
- Regression (Diabetes dataset) - commented out by default

Uncomment the sections you want to run.

### Custom Dataset Example

```python
from dqnn import DQNN
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Your custom data
X, y = your_data_loading_function()

# Split and preprocess
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# For classification
y_train = torch.LongTensor(y_train)  # or FloatTensor for binary
y_test = torch.LongTensor(y_test)

# For regression
# y_train = torch.FloatTensor(y_train)
# y_test = torch.FloatTensor(y_test)

# Initialize DQNN
dqnn = DQNN(
    X_train, y_train, X_test, y_test,
    task_type='classification'  # or 'regression'
)

# Train
episode_rewards = dqnn.train(
    episodes=50,              # Number of architecture search episodes
    max_steps_per_episode=10  # Steps per episode
)
```

### Task Types

DQNN automatically detects the task type:

1. **Binary Classification**: Automatically detected when there are 2 unique classes
   ```python
   dqnn = DQNN(X_train, y_train, X_test, y_test, task_type='classification')
   ```

2. **Multiclass Classification**: Automatically detected when there are >2 unique classes
   ```python
   dqnn = DQNN(X_train, y_train, X_test, y_test, task_type='classification')
   ```

3. **Regression**: Use for continuous target variables
   ```python
   dqnn = DQNN(X_train, y_train, X_test, y_test, task_type='regression')
   ```

## 📊 Dashboard

The Streamlit dashboard provides comprehensive visualization and analysis:

### Pages Overview

1. **Overview**: 
   - Total experiments
   - Best performance metrics
   - Architecture search evolution
   - Performance distribution

2. **Best Architecture**: 
   - Detailed architecture information
   - Hyperparameters
   - Visual representation
   - Generated PyTorch code

3. **Performance Analysis**: 
   - Hyperparameter impact
   - Correlation matrix
   - 3D performance visualization

4. **Hyperparameter Study**: 
   - Layer count analysis
   - Network size impact
   - Parameter ranges explored

5. **Detailed Results**: 
   - All experiment results
   - Sortable tables
   - Epoch-wise training progress

6. **AI Insights**: 
   - Model performance analysis
   - Architecture recommendations
   - Training optimization tips

7. **Export Results**: 
   - Download CSV/JSON data
   - Export best architecture
   - Data summaries

### Dashboard Features

- 📈 **Interactive Plots**: Zoom, pan, and explore your data
- 🎨 **Beautiful Visualizations**: Plotly-based charts
- 🔍 **Detailed Metrics**: Track every aspect of your experiments
- 💾 **Export Options**: Download results in multiple formats
- 🤖 **AI Analysis**: Get intelligent insights powered by Ollama

## 🤖 AI Insights with Ollama

The dashboard integrates with Ollama to provide AI-powered analysis:

### Setup Ollama

1. **Install Ollama** (see Installation section)

2. **Pull a Model**:
   ```bash
   # Recommended lightweight models
   ollama pull gemma2:2b      # Fast, 2B parameters
   ollama pull llama3.2:3b    # Balanced, 3B parameters
   
   # More powerful models
   ollama pull llama3.1:8b    # Advanced, 8B parameters
   ollama pull mixtral:8x7b   # Expert mixture, very capable
   ```

3. **Configure Model in Dashboard**:
   Edit `app.py`:
   ```python
   md = "gemma2:2b"  # Change to your preferred model
   ```

### AI Features

- **Performance Analysis**: Deep analysis of model performance across experiments
- **Architecture Recommendations**: Suggests improvements and alternative architectures
- **Training Optimization**: Provides tips for better training results

### Running Ollama

Make sure Ollama is running:
```bash
ollama serve
```

Or it will start automatically on most systems.

## 📁 Project Structure

```
dqnn/
│
├── dqnn.py                 # Core DQNN implementation
│   ├── DQNNetwork          # Q-Network for architecture search
│   ├── ArchitectureState   # Architecture representation
│   ├── ArchitectureGenerator  # Creates PyTorch models
│   ├── DQNNAgent          # Reinforcement learning agent
│   ├── ArchitectureActions # Architecture modification actions
│   ├── ModelTrainer       # Trains and evaluates architectures
│   ├── RewardCalculator   # Calculates rewards
│   └── DQNN               # Main orchestrator class
│
├── TorchTrack.py          # Experiment tracking system
│   ├── log()              # Log experiment data
│   ├── log_epoch()        # Log epoch-level data
│   └── clean_previous_data() # Clear old experiments
│
├── app.py             # Streamlit dashboard
│   ├── Overview           # Main metrics and visualizations
│   ├── Best Architecture  # Best model details
│   ├── Performance Analysis # Detailed analysis
│   ├── AI Insights        # Ollama-powered insights
│   └── Export Results     # Data export functionality
│
├── model.py               # Example usage file
│   ├── Binary Classification Example
│   ├── Multiclass Classification Example
│   └── Regression Example
│
├── data.json              # Experiment data (auto-generated)
├── best_architecture.json # Best architecture (auto-generated)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🧠 How It Works

### The DQNN Process

1. **Initialize**: Start with a simple architecture (one hidden layer)

2. **Explore**: The DQN agent explores the architecture space by:
   - Adding/removing layers
   - Modifying layer sizes
   - Changing activation functions
   - Adjusting hyperparameters (learning rate, batch size, epochs, dropout)

3. **Train**: Each architecture is trained and evaluated on your dataset

4. **Reward**: Calculate reward based on:
   - Performance (accuracy/R² score)
   - Model complexity (penalize too many parameters)
   - Training time (penalize long training)

5. **Learn**: The DQN agent learns which actions lead to better architectures

6. **Optimize**: Over multiple episodes, the agent discovers optimal architectures

7. **Track**: All experiments are logged with TorchTrack

8. **Visualize**: Results are displayed in the interactive dashboard

### Action Space

The agent can take 8 different actions:

| Action ID | Action | Description |
|-----------|--------|-------------|
| 0 | Add Layer | Add a new hidden layer |
| 1 | Remove Layer | Remove the last hidden layer |
| 2 | Modify Layer Size | Change the size of a random layer |
| 3 | Change Activation | Change activation function |
| 4 | Adjust Learning Rate | Modify the learning rate |
| 5 | Adjust Batch Size | Change the batch size |
| 6 | Adjust Epochs | Modify number of training epochs |
| 7 | Adjust Dropout | Change dropout rate |

### Reward Function

```
Reward = α × Performance - β × Complexity - γ × Time

Where:
- α = 1.0 (performance weight)
- β = 0.001 (complexity penalty)
- γ = 0.01 (time penalty)
```

## 📚 Examples

### Example 1: Binary Classification (Breast Cancer)

```python
from dqnn import DQNN
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = torch.FloatTensor(scaler.fit_transform(X_train))
X_test = torch.FloatTensor(scaler.transform(X_test))
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

dqnn = DQNN(X_train, y_train, X_test, y_test, task_type='classification')
dqnn.train(episodes=20, max_steps_per_episode=8)
```

### Example 2: Multiclass Classification (Iris)

```python
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = torch.FloatTensor(scaler.fit_transform(X_train))
X_test = torch.FloatTensor(scaler.transform(X_test))
y_train = torch.LongTensor(y_train)  # Long tensor for multiclass
y_test = torch.LongTensor(y_test)

dqnn = DQNN(X_train, y_train, X_test, y_test, task_type='classification')
dqnn.train(episodes=20, max_steps_per_episode=8)
```

### Example 3: Regression (Diabetes)

```python
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = torch.FloatTensor(scaler.fit_transform(X_train))
X_test = torch.FloatTensor(scaler.transform(X_test))
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

dqnn = DQNN(X_train, y_train, X_test, y_test, task_type='regression')
dqnn.train(episodes=20, max_steps_per_episode=8)
```

## 🔧 Configuration

### DQNN Parameters

```python
dqnn = DQNN(
    X_train,                  # Training features
    y_train,                  # Training labels
    X_test,                   # Test features
    y_test,                   # Test labels
    task_type='classification' # 'classification' or 'regression'
)

episode_rewards = dqnn.train(
    episodes=50,              # Number of search episodes
    max_steps_per_episode=10  # Steps per episode
)
```

### DQN Agent Configuration

Edit `dqnn.py` to modify agent parameters:

```python
class DQNNAgent:
    def __init__(self, ...):
        self.epsilon = 1.0           # Initial exploration rate
        self.epsilon_min = 0.01      # Minimum exploration rate
        self.epsilon_decay = 0.995   # Exploration decay rate
        self.learning_rate = 0.001   # DQN learning rate
        self.gamma = 0.95            # Discount factor
        self.batch_size = 32         # Batch size for replay
        self.target_update_freq = 100 # Target network update frequency
```

### Reward Weights

Adjust reward calculation in `dqnn.py`:

```python
reward_calculator = RewardCalculator(
    alpha=1.0,    # Performance weight
    beta=0.001,   # Complexity penalty
    gamma=0.01    # Time penalty
)
```

## 📊 Output Files

After training, DQNN generates:

- **`data.json`**: Complete experiment history with all trials
- **`best_architecture.json`**: Best architecture found with all details
- **Dashboard visualizations**: Available at `http://localhost:8501`

### Best Architecture JSON Structure

```json
{
    "layers": [128, 64, 32],
    "activations": ["relu", "relu", "relu"],
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 150,
    "dropout_rate": 0.2,
    "performance": 0.9825,
    "task_type": "binary_classification",
    "num_classes": null
}
```

## 🎓 Tips for Best Results

1. **Start Small**: Begin with fewer episodes to test your setup
   ```python
   dqnn.train(episodes=10, max_steps_per_episode=5)
   ```

2. **Increase Gradually**: Once stable, increase search depth
   ```python
   dqnn.train(episodes=50, max_steps_per_episode=10)
   ```

3. **Use GPU**: Training is much faster with CUDA
   ```python
   # DQNN automatically uses GPU if available
   print(torch.cuda.is_available())  # Check GPU availability
   ```

4. **Standardize Data**: Always standardize/normalize your features
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_train = scaler.fit_transform(X_train)
   X_test = scaler.transform(X_test)
   ```

5. **Monitor Progress**: Watch the console output during training

6. **Use the Dashboard**: Visualize results in real-time

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ImportError: No module named 'torch'`
```bash
pip install torch torchvision torchaudio
```

**Issue**: `ModuleNotFoundError: No module named 'TorchTrack'`
- Ensure `TorchTrack.py` is in the same directory as `dqnn.py`

**Issue**: Ollama connection error
```bash
# Start Ollama service
ollama serve

# Check if model is available
ollama list
```

**Issue**: Dashboard not loading
```bash
# Reinstall Streamlit
pip install --upgrade streamlit

# Clear cache
streamlit cache clear
```

**Issue**: CUDA out of memory
- Reduce batch size in the architecture
- Use fewer episodes or steps
- Close other GPU-intensive applications

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyTorch team for the deep learning framework
- Scikit-learn for the datasets and utilities
- Streamlit for the amazing dashboard framework
- Ollama for local AI capabilities

## 🌟 Star History

If you find this project useful, please consider giving it a star! ⭐
