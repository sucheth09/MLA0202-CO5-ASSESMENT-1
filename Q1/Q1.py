import numpy as np

grid = np.array([
    [0, 0, 0, -1, 0],
    [0, -1, 0, -1, 0],
    [0, 0, 0,  0, 0],
    [-1, 0, -1,  0, 0],
    [0, 0,  0,  0, 2]
])

actions = [(-1,0),(1,0),(0,-1),(0,1)]
symbols = ['U','D','L','R']
gamma = 0.9
V = np.zeros(grid.shape)

def move(state, action):
    r, c = state
    nr, nc = r + action[0], c + action[1]
    if nr < 0 or nr >= 5 or nc < 0 or nc >= 5 or grid[nr,nc] == -1:
        return state, -10
    if grid[nr,nc] == 2:
        return (nr,nc), 20
    return (nr,nc), -1

for _ in range(100):
    newV = V.copy()
    for r in range(5):
        for c in range(5):
            if grid[r,c] == -1 or grid[r,c] == 2:
                continue
            values = []
            for a in actions:
                s1, r1 = move((r,c), a)
                s2, r2 = move((r,c), actions[(actions.index(a)+1)%4])
                values.append(0.8*(r1 + gamma*V[s1]) + 0.2*(r2 + gamma*V[s2]))
            newV[r,c] = max(values)
    if np.max(abs(newV-V)) < 0.001:
        break
    V = newV

policy = np.full(grid.shape, ' ')
for r in range(5):
    for c in range(5):
        if grid[r,c] == -1:
            policy[r,c] = '#'
        elif grid[r,c] == 2:
            policy[r,c] = 'G'
        else:
            values = []
            for a in actions:
                s1, r1 = move((r,c), a)
                s2, r2 = move((r,c), actions[(actions.index(a)+1)%4])
                values.append(0.8*(r1 + gamma*V[s1]) + 0.2*(r2 + gamma*V[s2]))
            policy[r,c] = symbols[np.argmax(values)]

print("Value Function:")
print(np.round(V,2))
print("\nOptimal Policy:")
print(policy)
