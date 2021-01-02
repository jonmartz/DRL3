import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn import tree

# env_name = 'CartPole-v1'
# df = pd.read_csv('hist_%s.csv' % env_name)
# fig, ax = plt.subplots(figsize=(15, 4))
# ccp_alpha = 0.005
# model = DecisionTreeClassifier(random_state=1, ccp_alpha=ccp_alpha)

env_name = 'Acrobot-v1'
df = pd.read_csv('hist_%s.csv' % env_name)
df = df.loc[(df['step'] < 50) & (df['action'] != 1)]
fig, ax = plt.subplots(figsize=(13, 5))
ccp_alpha = 0.002
model = DecisionTreeClassifier(random_state=1, ccp_alpha=ccp_alpha)

# env_name = 'MountainCarContinuous-v0'
# df = pd.read_csv('hist_%s.csv' % env_name)[:10000]
# fig, ax = plt.subplots(figsize=(18, 6))
# ccp_alpha = 0.001
# model = DecisionTreeRegressor(random_state=1, ccp_alpha=ccp_alpha)

df = df.drop(columns=['ep', 'step'])
x = df.drop(columns='action')
y = df['action']

features = df.columns[:-1]

model.fit(x, y)
print('ccp_alpha: %s, acc: %.4f' % (ccp_alpha, model.score(x, y)))


annotations = tree.plot_tree(model, feature_names=features,
                             # class_names=class_names,
                             fontsize=10, filled=True, impurity=False, proportion=True, rounded=True)
plt.savefig('tree_%s.png' % env_name, bbox_inches='tight')
plt.show()
