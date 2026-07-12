"""
Actor-Critic Calculation Demo
For AI Engineering (Fundamental) Class, LU Lab., Myanmar
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import random
import math

# --- CONFIG ---
ROWS, COLS = 5, 5
CELL_SIZE = 80 
OFFSET = 25  # Space for X/Y axis labels
GOAL = (0, 4)
TRAPS = [(1, 3), (2, 3), (3, 3)]
WALLS = [(2, 1), (2, 2)]

# --- PURE PYTHON NEURAL NETWORK ---
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size, lr=0.05):
        self.W1 = [[random.uniform(-0.5, 0.5) for _ in range(input_size)] for _ in range(hidden_size)]
        self.b1 = [0.0 for _ in range(hidden_size)]
        self.W2 = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)] for _ in range(output_size)]
        self.b2 = [0.0 for _ in range(output_size)]
        self.lr = lr

    def relu(self, x):
        return max(0.0, x)

    def d_relu(self, x):
        return 1.0 if x > 0.0 else 0.0

    def forward(self, x):
        self.inp = x
        self.z1 = [sum(w*i for w, i in zip(weights, x)) + b for weights, b in zip(self.W1, self.b1)]
        self.a1 = [self.relu(z) for z in self.z1]
        self.z2 = [sum(w*a for w, a in zip(weights, self.a1)) + b for weights, b in zip(self.W2, self.b2)]
        return self.z2

    def backward(self, grad_output):
        grad_a1 = [0.0] * len(self.a1)
        for i in range(len(self.W2)):
            for j in range(len(self.W2[i])):
                grad_a1[j] += grad_output[i] * self.W2[i][j]
                self.W2[i][j] -= self.lr * grad_output[i] * self.a1[j]
            self.b2[i] -= self.lr * grad_output[i]

        grad_z1 = [ga * self.d_relu(z) for ga, z in zip(grad_a1, self.z1)]

        for i in range(len(self.W1)):
            for j in range(len(self.W1[i])):
                self.W1[i][j] -= self.lr * grad_z1[i] * self.inp[j]
            self.b1[i] -= self.lr * grad_z1[i]

# --- DQN ENGINE ---
class DQNEngine:
    def __init__(self):
        self.nn = SimpleNN(2, 16, 4, lr=0.05)
        self.actions = ["up", "down", "left", "right"]
        
        self.memory = []
        self.batch_size = 16
        self.max_memory = 500
        
        self.gamma = 0.9
        self.epsilon = 1.0        
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.95 
        
        self.episode_history = [] 
        self.step_history = [] 
        self.episode_count = 0

    def normalize(self, state):
        return [state[0] / float(ROWS), state[1] / float(COLS)]

    def get_q_values(self, state):
        return self.nn.forward(self.normalize(state))

    def generate_episode(self):
        state = (4, 0)
        path = []
        for _ in range(60): 
            if state in [GOAL] + TRAPS: break
            
            if random.random() < self.epsilon:
                action_idx = random.randint(0, 3)
            else:
                q_values = self.get_q_values(state)
                action_idx = q_values.index(max(q_values))
            
            action = self.actions[action_idx]
            r, c = state
            if action == "up": r = max(0, r-1)
            elif action == "down": r = min(ROWS-1, r+1)
            elif action == "left": c = max(0, c-1)
            elif action == "right": c = min(COLS-1, c+1)
            
            next_state = (r, c) if (r, c) not in WALLS else state
            done = next_state in [GOAL] + TRAPS
            reward = 1.0 if next_state == GOAL else (-1.0 if next_state in TRAPS else -0.04)
            
            self.memory.append((state, action_idx, reward, next_state, done))
            if len(self.memory) > self.max_memory:
                self.memory.pop(0)
                
            path.append((state, action, reward))
            state = next_state
            
        self.episode_history = path
        self.step_history.append(len(path))
        self.episode_count += 1
        return path

    def train_batch(self):
        if len(self.memory) < self.batch_size:
            return 0.0, "Not enough memory to train. Run more episodes!"
            
        batch = random.sample(self.memory, self.batch_size)
        total_loss = 0.0
        
        for state, action_idx, reward, next_state, done in batch:
            if done:
                target = reward
            else:
                next_q = self.get_q_values(next_state)
                target = reward + self.gamma * max(next_q)
                
            current_q = self.get_q_values(state)
            
            error = current_q[action_idx] - target
            total_loss += 0.5 * (error ** 2) 
            
            grad_output = [0.0] * 4
            grad_output[action_idx] = error
            self.nn.backward(grad_output)
            
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        avg_loss = total_loss / self.batch_size
        return avg_loss, f"Trained on batch 16. MSE Loss: {avg_loss:.4f}"

# --- GUI VISUALIZER ---
class DQNVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Deep Q-Network Lab (Pure Python)")
        self.root.geometry("1100x700")
        self.engine = DQNEngine()
        self.show_policy = False
        self.is_auto_training = False 
        
        paned = ttk.PanedWindow(root, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=5, pady=5)
        
        left_panel = ttk.Frame(paned)
        paned.add(left_panel, weight=1)
        self.canvas = tk.Canvas(left_panel, width=COLS*CELL_SIZE + OFFSET, height=ROWS*CELL_SIZE + OFFSET, bg="white")
        self.canvas.pack(pady=10)
        
        ctrl = ttk.Frame(left_panel)
        ctrl.pack(fill="x")
        ttk.Button(ctrl, text="1. Run Episode (Collect Data)", command=self.run_episode).pack(fill="x", pady=2)
        ttk.Button(ctrl, text="2. Train Network (Backprop)", command=self.run_learn).pack(fill="x", pady=2)
        
        self.btn_auto = ttk.Button(ctrl, text="▶ Start Auto-Train", command=self.toggle_auto_train)
        self.btn_auto.pack(fill="x", pady=2)
        
        ttk.Button(ctrl, text="Toggle Network Policy", command=self.toggle_policy).pack(fill="x", pady=2)
        ttk.Button(ctrl, text="3. Reset Brain & Memory", command=self.reset_learning).pack(fill="x", pady=2)
        
        right_panel = ttk.Frame(paned)
        paned.add(right_panel, weight=2)
        
        theory_frame = ttk.LabelFrame(right_panel, text="Theory: Neural Network & Replay")
        theory_frame.pack(fill="x", pady=5)
        theory_text = (
            "1. THE BRAIN (Pure Python MLP):\n"
            "   - Input: [x, y] -> Hidden: 16 (ReLU) -> Output: 4 Q-Values\n"
            "2. THE LOSS (MSE):\n"
            "   - L = 0.5 * [Q(s,a) - (R + γ * max Q(s',a'))]^2\n"
            "3. BACKPROPAGATION & REPLAY:\n"
            "   - Agent randomly samples past memory to prevent forgetting."
        )
        ttk.Label(theory_frame, text=theory_text, font=("Consolas", 10), justify="left", padding=10).pack(anchor="w")
        
        self.console = scrolledtext.ScrolledText(right_panel, font=("Consolas", 9), bg="#fdfdfd")
        self.console.pack(fill="both", expand=True)
        
        self.draw_grid()

    def toggle_policy(self):
        self.show_policy = not self.show_policy
        self.draw_grid()

    def toggle_auto_train(self):
        self.is_auto_training = not self.is_auto_training
        if self.is_auto_training:
            self.btn_auto.config(text="⏸ Stop Auto-Train")
            self.console.insert(tk.END, "\n*** AUTO-TRAINING STARTED ***\n")
            self.auto_train_loop()
        else:
            self.btn_auto.config(text="▶ Start Auto-Train")
            self.console.insert(tk.END, "\n*** AUTO-TRAINING PAUSED ***\n")
            self.console.see(tk.END)

    def auto_train_loop(self):
        if not self.is_auto_training: return
        
        self.run_episode()
        self.run_learn(print_details=False) 
        
        if len(self.engine.step_history) >= 5:
            recent_avg = sum(self.engine.step_history[-5:]) / 5.0
            if self.engine.epsilon <= self.engine.epsilon_min and recent_avg < 12.0:
                self.is_auto_training = False
                self.btn_auto.config(text="▶ Start Auto-Train")
                self.console.insert(tk.END, f"\n✅ CONVERGENCE REACHED!\nAgent solved last 5 episodes in {recent_avg:.1f} steps on average.\n")
                self.console.see(tk.END)
                return

        self.root.after(75, self.auto_train_loop)

    def run_episode(self):
        path = self.engine.generate_episode()
        self.draw_grid()
        self.console.insert(tk.END, f"Episode {self.engine.episode_count} | Steps: {len(path)} | Mem: {len(self.engine.memory)}/{self.engine.max_memory} | Eps: {self.engine.epsilon:.2f}\n")
        self.console.see(tk.END)

    def run_learn(self, print_details=True):
        log = "--- BATCH TRAINING TRIGGERED ---\n"
        for i in range(5):
            loss, msg = self.engine.train_batch()
            if isinstance(msg, str) and "Not enough" in msg:
                if print_details: log += msg + "\n"
                break
            if print_details: log += f"  Batch {i+1} | {msg}\n"
            
        if print_details:
            log += f"  Current Epsilon (Exploration Rate): {self.engine.epsilon:.3f}\n\n"
            self.console.insert(tk.END, log)
            self.console.see(tk.END)
        self.draw_grid()

    def reset_learning(self):
        self.engine = DQNEngine() 
        self.is_auto_training = False
        self.btn_auto.config(text="▶ Start Auto-Train")
        self.console.delete("1.0", tk.END)
        self.console.insert(tk.END, "Brain and Memory reset complete.\n")
        self.draw_grid()

    def draw_grid(self):
        self.canvas.delete("all")
        
        # --- AXIS LABELS ---
        for c in range(COLS):
            self.canvas.create_text(c * CELL_SIZE + OFFSET + CELL_SIZE/2, OFFSET/2, 
                                    text=str(c), font=("Arial", 8, "bold"), fill="gray")
        for r in range(ROWS):
            self.canvas.create_text(OFFSET/2, r * CELL_SIZE + OFFSET + CELL_SIZE/2, 
                                    text=str(r), font=("Arial", 8, "bold"), fill="gray")
        
        path = self.engine.episode_history
        path_map = {step[0]: i+1 for i, step in enumerate(path)}
        start_state = path[0][0] if path else None
        end_state = path[-1][0] if path else None

        for r in range(ROWS):
            for c in range(COLS):
                state = (r, c)
                x, y = c * CELL_SIZE + OFFSET, r * CELL_SIZE + OFFSET
                
                q_values = self.engine.get_q_values(state)
                max_q = max(q_values)
                
                bg = "white"
                if state in WALLS: bg = "#333"
                elif state == GOAL: bg = "#d1e7dd"
                elif state in TRAPS: bg = "#f8d7da"
                elif max_q > 0: 
                    intensity = max(0, 255 - int(max_q * 150))
                    bg = f"#{intensity:02x}ff{intensity:02x}"
                elif max_q < 0: 
                    intensity = max(0, 255 - int(abs(max_q) * 100))
                    bg = f"#ff{intensity:02x}{intensity:02x}"
                
                self.canvas.create_rectangle(x, y, x+CELL_SIZE, y+CELL_SIZE, fill=bg, outline="#eee")
                
                if state not in WALLS and state not in [GOAL] + TRAPS:
                    self.canvas.create_text(x+CELL_SIZE/2, y+CELL_SIZE/2, text=f"{max_q:.2f}", font=("Arial", 10, "bold"), fill="black")

                if state in path_map:
                    self.canvas.create_oval(x+2, y+2, x+20, y+20, fill="#444", outline="white")
                    self.canvas.create_text(x+11, y+11, text=str(path_map[state]), font=("Arial", 8, "bold"), fill="white")
                    if state == start_state: self.canvas.create_text(x+CELL_SIZE-12, y+12, text="[S]", font=("Arial", 8, "bold"), fill="blue")
                    if state == end_state: self.canvas.create_text(x+CELL_SIZE-12, y+12, text="[E]", font=("Arial", 8, "bold"), fill="red")

                if self.show_policy and state not in WALLS and state not in [GOAL] + TRAPS:
                    best_act_idx = q_values.index(max_q)
                    act_name = self.engine.actions[best_act_idx]
                    arr = {"up": (0, -25), "down": (0, 25), "left": (-25, 0), "right": (25, 0)}
                    dx, dy = arr[act_name]
                    cx, cy = x + CELL_SIZE/2, y + CELL_SIZE/2
                    self.canvas.create_line(cx+(dx*0.4), cy+(dy*0.4), cx+dx, cy+dy, arrow=tk.LAST, width=2, fill="#555")

if __name__ == "__main__":
    root = tk.Tk()
    app = DQNVisualizer(root)
    root.mainloop()