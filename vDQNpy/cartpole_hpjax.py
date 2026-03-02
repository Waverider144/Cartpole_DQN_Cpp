import jax
import jax.numpy as jnp
from jax import random, jit, vmap
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
import gymnax
import numpy as np
import time
from PIL import Image
from functools import partial

# --- 1. 模型定义 ---
class DQNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.action_dim)(x)
        return x

# --- 2. 核心训练与环境交互算子 ---
@jit
def train_step(state, target_params, batch, gamma):
    def loss_fn(params):
        s, a, r, ns, d = batch
        q_values = state.apply_fn({'params': params}, s)
        current_q = jnp.take_along_axis(q_values, a, axis=1)
        
        next_q_values = state.apply_fn({'params': target_params}, ns)
        next_q = jnp.max(next_q_values, axis=1, keepdims=True)
        target_q = r + (gamma * next_q * (1 - d))
        return jnp.mean((current_q - target_q) ** 2)

    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)

# 使用 vmap 实现并行环境推断
@partial(jit, static_argnames=("action_dim"))
def get_action(params, obs, rng, eps, action_dim):
    rng_eps, rng_act = random.split(rng)
    
    # 探索动作
    random_actions = random.randint(rng_act, (obs.shape[0],), 0, action_dim)
    
    # 利用动作
    q_values = model.apply({'params': params}, obs)
    greedy_actions = jnp.argmax(q_values, axis=1)
    
    # 根据 epsilon 选择
    chose_random = random.uniform(rng_eps, (obs.shape[0],)) < eps
    return jnp.where(chose_random, random_actions, greedy_actions)

# --- 3. 结果保存与可视化 (保留你的逻辑) ---
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

def save_gym_gif(model_def, params, filename="cartpole_gymnax.gif"):
    import gymnasium as gym  # 渲染仍需原生 gym
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    state, _ = env.reset()
    frames = []
    for _ in range(500):
        frames.append(Image.fromarray(env.render()))
        obs_jax = jnp.array(state[None, :])
        q_vals = model_def.apply({'params': params}, obs_jax)
        action = int(jnp.argmax(q_vals))
        state, _, term, trunc, _ = env.step(action)
        if term or trunc: break
    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=20, loop=0)
    env.close()

# --- 4. 训练主循环 ---
if __name__ == "__main__":
    CONFIG = {
        "GAMMA": 0.99, "LR": 1e-3, "BATCH_SIZE": 128, 
        "MEM_CAP": 20000, "NUM_ENVS": 16, "TARGET_FREQ": 500,
        "TOTAL_EPISODES": 500, "EPS_START": 1.0, "EPS_END": 0.05, "EPS_DECAY": 5000
    }
    
    # 环境初始化 (Gymnax)
    rng = random.PRNGKey(42)
    rng, env_rng, model_rng = random.split(rng, 3)
    env, env_params = gymnax.make("CartPole-v1")
    
    # 向量化初始化环境
    v_reset = vmap(env.reset, in_axes=(0, None))
    v_step = vmap(env.step, in_axes=(0, 0, 0, None))
    
    # 模型初始化
    model = DQNetwork(action_dim=env.num_actions)
    params = model.init(model_rng, jnp.ones((1, 4)))['params']
    tx = optax.adamw(learning_rate=CONFIG["LR"])
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    target_params = params

    # GPU Replay Buffer (使用 JAX 数组)
    class JAXBuffer:
        def __init__(self, capacity, s_dim):
            self.s = jnp.zeros((capacity, s_dim))
            self.ns = jnp.zeros((capacity, s_dim))
            self.a = jnp.zeros((capacity, 1), dtype=jnp.int32)
            self.r = jnp.zeros((capacity, 1))
            self.d = jnp.zeros((capacity, 1))
            self.ptr, self.size, self.capacity = 0, 0, capacity

        def push(self, s, a, r, ns, d):
            num = s.shape[0]
            indices = (jnp.arange(num) + self.ptr) % self.capacity
            self.s = self.s.at[indices].set(s)
            self.ns = self.ns.at[indices].set(ns)
            self.a = self.a.at[indices].set(a[:, None])
            self.r = self.r.at[indices].set(r[:, None])
            self.d = self.d.at[indices].set(d[:, None])
            self.ptr = (self.ptr + num) % self.capacity
            self.size = jnp.minimum(self.size + num, self.capacity)

        def sample(self, batch_size, rng):
            idx = random.randint(rng, (batch_size,), 0, self.size)
            return (self.s[idx], self.a[idx], self.r[idx], self.ns[idx], self.d[idx])

    buffer = JAXBuffer(CONFIG["MEM_CAP"], 4)
    
    # 运行变量
    ep_durations, ep_times = [], []
    env_step_counts = np.zeros(CONFIG["NUM_ENVS"])
    start_time = time.time()
    
    # 初始状态
    rng, reset_rng = random.split(rng)
    reset_rngs = random.split(reset_rng, CONFIG["NUM_ENVS"])
    obs, env_state = v_reset(reset_rngs, env_params)
    
    total_steps = 0
    print(f"All JAX/Gymnax Environment Initialized. Speed will be insane.")

    while len(ep_durations) < CONFIG["TOTAL_EPISODES"]:
        eps = CONFIG["EPS_END"] + (CONFIG["EPS_START"]-CONFIG["EPS_END"]) * \
              jnp.exp(-1. * total_steps / CONFIG["EPS_DECAY"])
        
        # 选择动作
        rng, act_rng = random.split(rng)
        actions = get_action(state.params, obs, act_rng, eps, env.num_actions)
        
        # 环境交互
        rng, step_rng = random.split(rng)
        step_rngs = random.split(step_rng, CONFIG["NUM_ENVS"])
        next_obs, next_env_state, rewards, dones, info = v_step(step_rngs, env_state, actions, env_params)
        
        # 存入 Buffer
        buffer.push(obs, actions, rewards, next_obs, dones)
        
        # 更新统计
        env_step_counts += 1
        for i in range(CONFIG["NUM_ENVS"]):
            if dones[i]:
                ep_durations.append(int(env_step_counts[i]))
                ep_times.append(time.time() - start_time)
                env_step_counts[i] = 0
                if len(ep_durations) % 10 == 0:
                    print(f"Ep: {len(ep_durations)} | Avg(10): {np.mean(ep_durations[-10:]):.1f}")

        obs = next_obs
        env_state = next_env_state
        total_steps += CONFIG["NUM_ENVS"]

        # 训练更新
        if buffer.size > CONFIG["BATCH_SIZE"]:
            rng, sample_rng = random.split(rng)
            batch = buffer.sample(CONFIG["BATCH_SIZE"], sample_rng)
            state = train_step(state, target_params, batch, CONFIG["GAMMA"])

        if total_steps % CONFIG["TARGET_FREQ"] == 0:
            target_params = state.params

    print("\nTraining Complete.")
    save_results_to_txt(ep_durations, ep_times, CONFIG, "jax_gymnax_results.txt")
    save_gym_gif(model, state.params)