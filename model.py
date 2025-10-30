# A set of sample self contained examples from sklearn native examples

from dqnn import *

# Example 1: Binary Classification - Breast Cancer

print("BINARY CLASSIFICATION - Breast Cancer Dataset")

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

dqnn = DQNN(X_train, y_train, X_test, y_test, task_type='classification')
episode_rewards = dqnn.train(episodes=5, max_steps_per_episode=3)

# Example 2: Multiclass Classification - Iris

# print("MULTICLASS CLASSIFICATION - Iris Dataset")

# from sklearn.datasets import load_iris

# X, y = load_iris(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# X_train = torch.FloatTensor(X_train)
# y_train = torch.LongTensor(y_train)
# X_test = torch.FloatTensor(X_test)
# y_test = torch.LongTensor(y_test)

# dqnn_mc = DQNN(X_train, y_train, X_test, y_test, task_type='classification')
# episode_rewards_mc = dqnn_mc.train(episodes=5, max_steps_per_episode=3)

# # Example 3: Regression - Diabetes

# print("REGRESSION - Diabetes Dataset")

# from sklearn.datasets import load_diabetes

# X, y = load_diabetes(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# X_train = torch.FloatTensor(X_train)
# y_train = torch.FloatTensor(y_train)
# X_test = torch.FloatTensor(X_test)
# y_test = torch.FloatTensor(y_test)

# dqnn_reg = DQNN(X_train, y_train, X_test, y_test, task_type='regression')
# episode_rewards_reg = dqnn_reg.train(episodes=5, max_steps_per_episode=3)


# print("All tasks completed! Check the Streamlit dashboard for results.")
