import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import random
import sys
from PIL import Image

# --- 1. 模型定义 ---
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.layers(x)

def _update_network(policy_net, target_net, optimizer, states, actions, rewards, next_states, dones, gamma):
    current_q = policy_net(states).gather(1, actions)
    with torch.no_grad():
        next_q = target_net(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + (gamma * next_q * (1 - dones))
    
    loss = nn.functional.mse_loss(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()
    return loss

# --- 2. 向量化 Buffer (优化版) ---
class VectorizedReplayBuffer:
    def __init__(self, capacity, state_dim, num_envs, device):
        self.capacity = capacity
        self.num_envs = num_envs
        self.device = device
        self.ptr = 0
        self.size = 0
        self.states = torch.zeros((capacity, *state_dim), device=device)
        self.next_states = torch.zeros((capacity, *state_dim), device=device)
        self.actions = torch.zeros((capacity, 1), dtype=torch.long, device=device)
        self.rewards = torch.zeros((capacity, 1), device=device)
        self.dones = torch.zeros((capacity, 1), device=device)

    def push(self, s, a, r, ns, d):
        for i in range(self.num_envs):
            idx = self.ptr
            self.states[idx] = torch.as_tensor(s[i], device=self.device)
            self.next_states[idx] = torch.as_tensor(ns[i], device=self.device)
            self.actions[idx] = torch.as_tensor(a[i], device=self.device).unsqueeze(0)
            self.rewards[idx] = torch.as_tensor(r[i], device=self.device).unsqueeze(0)
            self.dones[idx] = torch.as_tensor(d[i], device=self.device, dtype=torch.float32).unsqueeze(0)
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, batch_size)
        return (self.states[indices], self.actions[indices], 
                self.rewards[indices], self.next_states[indices], self.dones[indices])

# --- 3. 结果保存与可视化 ---
def save_results_to_txt(episode_data, episode_times, params, filename):
    print(f"\nSaving results to {filename}...")
    with open(filename, 'w') as f:
        f.write("# --- Hyperparameters ---\n")
        for key, value in params.items():
            f.write(f"# {key}: {value}\n")
        f.write("# -----------------------\n\n")
        f.write("Episode\tLifespan (t+1)\tPhysical Time (s)\n") 
        for i, (duration, time_spent) in enumerate(zip(episode_data, episode_times)):
            f.write(f"{i+1}\t{duration}\t{time_spent:.4f}\n")
    print("Saving complete.")

def save_gym_gif(policy_net, device, filename="cartpole_v2.gif"):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    state, _ = env.reset()
    frames = []
    for _ in range(500):
        frames.append(Image.fromarray(env.render()))
        st = torch.as_tensor(state, device=device).float().unsqueeze(0)
        with torch.no_grad():
            action = policy_net(st).max(1)[1].item()
        state, _, term, trunc, _ = env.step(action)
        if term or trunc: break
    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=20, loop=0)
    env.close()

# --- 4. 训练主循环 ---
if __name__ == "__main__":
    CONFIG = {
        "GAMMA": 0.99, "LR": 1e-3, "BATCH_SIZE": 128, 
        "MEM_CAP": 20000, "NUM_ENVS": 8, "TARGET_FREQ": 500,
        "TOTAL_EPISODES": 500, "EPS_START": 1.0, "EPS_END": 0.05, "EPS_DECAY": 5000
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    envs = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(CONFIG["NUM_ENVS"])])
    s_dim = envs.single_observation_space.shape[0]
    a_dim = envs.single_action_space.n

    policy_net = DQN(s_dim, a_dim).to(device)
    target_net = DQN(s_dim, a_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    if hasattr(torch, 'compile'):
        optimized_train = torch.compile(_update_network)
    else:
        optimized_train = _update_network
    
    optimizer = optim.AdamW(policy_net.parameters(), lr=CONFIG["LR"])
    memory = VectorizedReplayBuffer(CONFIG["MEM_CAP"], (s_dim,), CONFIG["NUM_ENVS"], device)

    # 数据记录容器
    ep_durations = []
    ep_times = []
    env_step_counts = np.zeros(CONFIG["NUM_ENVS"])
    
    start_time = time.time()  # 记录物理开始时间
    states, _ = envs.reset()
    total_steps = 0
    
    print(f"Device: {device} | Starting Training...")

    while len(ep_durations) < CONFIG["TOTAL_EPISODES"]:
        eps = CONFIG["EPS_END"] + (CONFIG["EPS_START"]-CONFIG["EPS_END"]) * \
              np.exp(-1. * total_steps / CONFIG["EPS_DECAY"])
        
        if random.random() < eps:
            actions = envs.action_space.sample()
        else:
            with torch.no_grad():
                st_tensor = torch.as_tensor(states, device=device, dtype=torch.float32)
                actions = policy_net(st_tensor).max(1)[1].cpu().numpy()

        next_states, rewards, terms, truncs, _ = envs.step(actions)
        dones = terms | truncs
        
        memory.push(states, actions, rewards, next_states, dones)
        
        env_step_counts += 1
        new_done_flag = False
        
        for i, done in enumerate(dones):
            if done:
                # 记录该 episode 的时长和当前的物理耗时
                ep_durations.append(int(env_step_counts[i]))
                ep_times.append(time.time() - start_time)
                
                env_step_counts[i] = 0
                new_done_flag = True
        
        # 实时输出进度
        if new_done_flag and len(ep_durations) % 10 == 0:
            avg_rew = np.mean(ep_durations[-10:])
            print(f"Ep: {len(ep_durations)} | Avg(10): {avg_rew:.1f} | Live Steps: {env_step_counts[-1]}")

        states = next_states
        total_steps += CONFIG["NUM_ENVS"]

        if memory.size > CONFIG["BATCH_SIZE"]:
            b_s, b_a, b_r, b_ns, b_d = memory.sample(CONFIG["BATCH_SIZE"])
            _ = optimized_train(policy_net, target_net, optimizer, b_s, b_a, b_r, b_ns, b_d, CONFIG["GAMMA"])

        if total_steps % CONFIG["TARGET_FREQ"] == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print("\nTraining Finished.")
    envs.close()
    
    # 保存结果
    save_results_to_txt(ep_durations, ep_times, CONFIG, "training_log.txt")
    save_gym_gif(policy_net, device)