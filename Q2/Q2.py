import numpy as np

np.random.seed(42)

true_probabilities = [0.05, 0.10, 0.07, 0.15, 0.12]
n_ads = 5
n_rounds = 10000

counts = np.zeros(n_ads)
rewards = np.zeros(n_ads)
total_reward = 0

for i in range(n_ads):
    reward = np.random.binomial(1, true_probabilities[i])
    counts[i] += 1
    rewards[i] += reward
    total_reward += reward

for t in range(n_ads, n_rounds):
    mean_rewards = rewards / counts
    confidence = np.sqrt(2 * np.log(t + 1) / counts)
    ucb = mean_rewards + confidence

    ad = np.argmax(ucb)

    reward = np.random.binomial(1, true_probabilities[ad])

    counts[ad] += 1
    rewards[ad] += reward
    total_reward += reward

print("Advertisement Selection Counts:")
print(counts.astype(int))

print("\nEstimated Click Probabilities:")
print(np.round(rewards / counts, 3))

print("\nTotal Clicks:", int(total_reward))
print("Best Advertisement:", np.argmax(rewards / counts) + 1)
