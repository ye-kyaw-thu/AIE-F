"""
Actor-Critic Calculation Demo
For AI Engineering (Fundamental) Class, LU Lab., Myanmar
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import random
import time
import sys

# --- THEME ENGINE DEFINITIONS ---
THEMES = {
    "light": {
        "bg": "#f4f6f9", "card": "#ffffff", "text": "#212529", "accent": "#007bff",
        "success": "#28a745", "warn": "#dc3545", "wall": "#6c757d", "grid_line": "#dee2e6"
    },
    "dark": {
        "bg": "#1e1e24", "card": "#2a2b36", "text": "#f8f9fa", "accent": "#3a86ff",
        "success": "#2ec4b6", "warn": "#ff006e", "wall": "#4a4e69", "grid_line": "#3d3f4d"
    },
    "solarized": {
        "bg": "#fdf6e3", "card": "#eee8d5", "text": "#586e75", "accent": "#b58900",
        "success": "#859900", "warn": "#dc322f", "wall": "#93a1a1", "grid_line": "#cb4b16"
    },
    "oceanic": {
        "bg": "#0f172a", "card": "#1e293b", "text": "#f8fafc", "accent": "#38bdf8",
        "success": "#34d399", "warn": "#f87171", "wall": "#475569", "grid_line": "#334155"
    },
    "emerald": {
        "bg": "#f0fdf4", "card": "#ffffff", "text": "#166534", "accent": "#15803d",
        "success": "#16a34a", "warn": "#b91c1c", "wall": "#78716c", "grid_line": "#bbf7d0"
    }
}

# Parse command line arguments gracefully
SELECTED_THEME = "light"
TRAIN_EPISODES = 50  # Default value if argument is omitted
for arg in sys.argv:
    if arg.startswith("--theme="):
        t_val = arg.split("=")[1].lower()
        if t_val in THEMES: SELECTED_THEME = t_val
    elif arg in ["-t", "--theme"]:
        try:
            idx = sys.argv.index(arg)
            t_val = sys.argv[idx + 1].lower()
            if t_val in THEMES: SELECTED_THEME = t_val
        except (IndexError, ValueError):
            pass
    elif arg.startswith("--episodes="):
        try:
            TRAIN_EPISODES = int(arg.split("=")[1])
        except ValueError:
            pass

C = THEMES[SELECTED_THEME]

# --- ENVIRONMENT CONFIGURATION ---
ROWS = 6
COLS = 9
CELL_SIZE = 44  
START = (3, 4)
GOAL = (0, 8)
DEFAULT_BLOCKS = [(1, 2), (2, 2), (3, 2), (4, 2), (1, 6), (2, 6), (3, 6), (4, 6), (1, 3), (1, 4), (1, 5)]
ACTIONS = ["left", "up", "right", "down"]

class Environment:
    def __init__(self):
        self.rows = ROWS
        self.cols = COLS
        self.start = START
        self.goal = GOAL
        self.blocks = set(DEFAULT_BLOCKS)
        self.state = START
        self.done = False

    def reset(self):
        self.state = self.start
        self.done = False
        return self.state

    def step(self, action):
        if self.done:
            return self.state, 0, True

        r, c = self.state
        if action == "left": c -= 1
        elif action == "right": c += 1
        elif action == "up": r -= 1
        elif action == "down": r += 1

        if 0 <= r < self.rows and 0 <= c < self.cols:
            if (r, c) not in self.blocks:
                self.state = (r, c)

        self.done = (self.state == self.goal)
        reward = 1.0 if self.done else 0.0
        return self.state, reward, self.done

class GeneralAgent:
    def __init__(self, env):
        self.env = env
        self.actions = ACTIONS
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.1
        self.planning_steps = 10
        self.kappa = 0.001 
        self.current_method = "Dyna-Q"
        
        # Primary value matrices
        self.Q = {}  
        self.Q1 = {}  # Specifically for D Q-Learning
        self.Q2 = {}  # Specifically for D Q-Learning
        
        self.model = {}  
        self.visited_states = set()
        self.time_elapsed = {}  
        self.initialize_time_tracking()

    def initialize_time_tracking(self):
        self.time_elapsed.clear()
        for r in range(ROWS):
            for c in range(COLS):
                for a in ACTIONS:
                    self.time_elapsed[((r, c), a)] = 0

    def get_q(self, state, action):
        if self.current_method == "D Q-Learning":
            return (self.Q1.get((state, action), 0.0) + self.Q2.get((state, action), 0.0)) / 2.0
        return self.Q.get((state, action), 0.0)

    def choose_action(self, state, exploit_only=False):
        # Dyna-Q+: Inject the curiosity exploration bonus directly at action-selection time
        if self.current_method == "Dyna-Q+" and not exploit_only:
            q_vals = [self.get_q(state, a) + self.kappa * np.sqrt(self.time_elapsed.get((state, a), 0)) for a in self.actions]
            max_q = max(q_vals)
            best_actions = [a for a, q in zip(self.actions, q_vals) if abs(q - max_q) < 1e-5]
            return random.choice(best_actions)
            
        if not exploit_only and random.random() < self.epsilon:
            return random.choice(self.actions)
        
        q_vals = [self.get_q(state, a) for a in self.actions]
        max_q = max(q_vals)
        best_actions = [a for a, q in zip(self.actions, q_vals) if abs(q - max_q) < 1e-5]
        return random.choice(best_actions)

    def learn_transition(self, state, action, reward, next_state, next_action=None, method="Dyna-Q"):
        self.current_method = method
        
        # Explicit time tracker increment for Dyna-Q+ features
        for pair in self.time_elapsed:
            if pair == (state, action):
                self.time_elapsed[pair] = 0
            else:
                self.time_elapsed[pair] += 1

        self.visited_states.add(state)
        self.model[(state, action)] = (reward, next_state)

        # --- ALGORITHM STRATEGY ROUTING CORES ---
        if method == "SARSA":
            next_q = self.get_q(next_state, next_action) if next_action else 0.0
            td_target = reward + self.gamma * next_q
            current_q = self.get_q(state, action)
            self.Q[(state, action)] = current_q + self.alpha * (td_target - current_q)

        elif method == "Q-Learning":
            max_next_q = max([self.get_q(next_state, a) for a in self.actions])
            td_target = reward + self.gamma * max_next_q
            current_q = self.get_q(state, action)
            self.Q[(state, action)] = current_q + self.alpha * (td_target - current_q)

        elif method == "Exp SARSA":
            q_vals = [self.get_q(next_state, a) for a in self.actions]
            max_next_q = max(q_vals)
            best_actions = [a for a, q in zip(self.actions, q_vals) if abs(q - max_next_q) < 1e-5]
            
            expected_v = 0.0
            n_actions = len(self.actions)
            n_greedy = len(best_actions)
            
            for a, q in zip(self.actions, q_vals):
                prob = self.epsilon / n_actions
                if a in best_actions:
                    prob += (1.0 - self.epsilon) / n_greedy
                expected_v += prob * q
                
            td_target = reward + self.gamma * expected_v
            current_q = self.get_q(state, action)
            self.Q[(state, action)] = current_q + self.alpha * (td_target - current_q)

        elif method == "D Q-Learning":
            if random.random() < 0.5:
                q1_vals = [self.Q1.get((next_state, a), 0.0) for a in self.actions]
                max_q1 = max(q1_vals)
                best_q1_acts = [a for a, q in zip(self.actions, q1_vals) if abs(q - max_q1) < 1e-5]
                chosen_best_a = random.choice(best_q1_acts)
                
                td_target = reward + self.gamma * self.Q2.get((next_state, chosen_best_a), 0.0)
                current_q1 = self.Q1.get((state, action), 0.0)
                self.Q1[(state, action)] = current_q1 + self.alpha * (td_target - current_q1)
            else:
                q2_vals = [self.Q2.get((next_state, a), 0.0) for a in self.actions]
                max_q2 = max(q2_vals)
                best_q2_acts = [a for a, q in zip(self.actions, q2_vals) if abs(q - max_q2) < 1e-5]
                chosen_best_a = random.choice(best_q2_acts)
                
                td_target = reward + self.gamma * self.Q1.get((next_state, chosen_best_a), 0.0)
                current_q2 = self.Q2.get((state, action), 0.0)
                self.Q2[(state, action)] = current_q2 + self.alpha * (td_target - current_q2)

        elif method in ["Dyna-Q", "Dyna-Q+"]:
            max_next_q = max([self.get_q(next_state, a) for a in self.actions])
            td_target = reward + self.gamma * max_next_q
            current_q = self.get_q(state, action)
            self.Q[(state, action)] = current_q + self.alpha * (td_target - current_q)

            if len(self.visited_states) > 0:
                for _ in range(self.planning_steps):
                    sim_state = random.choice(list(self.visited_states))
                    sim_action = random.choice(self.actions)
                    
                    if method == "Dyna-Q":
                        if (sim_state, sim_action) in self.model:
                            sim_reward, sim_next_state = self.model[(sim_state, sim_action)]
                            sim_max_next_q = max([self.get_q(sim_next_state, a) for a in self.actions])
                            sim_target = sim_reward + self.gamma * sim_max_next_q
                            sim_current_q = self.get_q(sim_state, sim_action)
                            self.Q[(sim_state, sim_action)] = sim_current_q + self.alpha * (sim_target - sim_current_q)
                            
                    elif method == "Dyna-Q+":
                        # Standardized tabular planning structure (avoids self-loop inflation)
                        if (sim_state, sim_action) in self.model:
                            sim_reward, sim_next_state = self.model[(sim_state, sim_action)]
                        else:
                            sim_reward, sim_next_state = 0.0, sim_state
                        
                        sim_max_next_q = max([self.get_q(sim_next_state, a) for a in self.actions])
                        sim_target = sim_reward + self.gamma * sim_max_next_q
                        sim_current_q = self.get_q(sim_state, sim_action)
                        self.Q[(sim_state, sim_action)] = sim_current_q + self.alpha * (sim_target - sim_current_q)

    def reset_knowledge(self):
        self.Q.clear()
        self.Q1.clear()
        self.Q2.clear()
        self.model.clear()
        self.visited_states.clear()
        self.initialize_time_tracking()


class EducationalSuite:
    def __init__(self, root):
        self.root = root
        self.root.title(f"RL Laboratory Suite [Theme: {SELECTED_THEME.upper()}]")
        self.root.configure(bg=C["bg"])
        self.root.geometry("980x610")
        
        self.env = Environment()
        self.agent = GeneralAgent(self.env)
        
        self.eval_history = {
            "Q-Learning": [], "SARSA": [], "Exp SARSA": [], 
            "D Q-Learning": [], "Dyna-Q": [], "Dyna-Q+": []
        }
        self.algo_colors = {
            "Q-Learning": C["warn"], "SARSA": "#fd7e14", "Exp SARSA": "#198754", 
            "D Q-Learning": "#0dcaf0", "Dyna-Q": C["accent"], "Dyna-Q+": "#6f42c1"
        }
        
        self.is_running = False
        self.anim_state = None
        self.anim_action = None
        self.anim_steps = 0
        
        self.build_ui()
        self.draw_maze()
        self.draw_evaluation_graph()
        self.log(f"Successfully mounted laboratory with {SELECTED_THEME} theme parameters.")

    def build_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', background=C["bg"], foreground=C["text"])
        style.configure('TLabel', background=C["bg"], foreground=C["text"], font=("Segoe UI", 9))
        style.configure('TFrame', background=C["bg"])
        style.configure('Card.TFrame', background=C["card"], borderwidth=1, relief="solid", bordercolor=C["grid_line"])
        
        self.root.columnconfigure(0, weight=2)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # --- LEFT PANEL ---
        left_panel = ttk.Frame(self.root, padding=8)
        left_panel.grid(row=0, column=0, sticky="nsew")
        
        title_card = ttk.Frame(left_panel, style="Card.TFrame", padding=8)
        title_card.pack(fill="x", pady=(0, 6))
        tk.Label(title_card, text="🎓 Reinforcement Learning Laboratory for AI Engineering Students", font=("Segoe UI", 14, "bold"), fg=C["accent"], bg=C["card"]).pack(anchor="w")
        tk.Label(title_card, text="Click/Drag Left Mouse Button to place walls. Right click to remove walls.", font=("Segoe UI", 9, "italic"), fg=C["text"], bg=C["card"]).pack(anchor="w")

        canvas_card = ttk.Frame(left_panel, style="Card.TFrame", padding=6)
        canvas_card.pack(fill="both", expand=True, pady=(0, 6))
        
        self.canvas = tk.Canvas(canvas_card, width=COLS*CELL_SIZE, height=ROWS*CELL_SIZE, bg=C["card"], highlightthickness=0)
        self.canvas.pack(anchor="center", pady=2)
        
        self.canvas.bind("<B1-Motion>", self.canvas_paint_wall)
        self.canvas.bind("<Button-1>", self.canvas_paint_wall)
        self.canvas.bind("<B3-Motion>", self.canvas_clear_wall)
        self.canvas.bind("<Button-3>", self.canvas_clear_wall)

        graph_card = ttk.Frame(left_panel, style="Card.TFrame", padding=8)
        graph_card.pack(fill="x", pady=0)
        tk.Label(graph_card, text="📈 Live Performance Curve (Steps per Episode - Log Scale)", font=("Segoe UI", 10, "bold"), fg=C["text"], bg=C["card"]).pack(anchor="w", pady=(0, 2))
        
        self.graph_canvas = tk.Canvas(graph_card, height=115, bg=C["card"], highlightthickness=0)
        self.graph_canvas.pack(fill="x", expand=True)
        
        # --- RIGHT PANEL ---
        right_panel = ttk.Frame(self.root, padding=8)
        right_panel.grid(row=0, column=1, sticky="nsew")
        
        cfg_card = ttk.Frame(right_panel, style="Card.TFrame", padding=8)
        cfg_card.pack(fill="x", pady=(0, 6))
        
        tk.Label(cfg_card, text="Model Strategy Selection", font=("Segoe UI", 10, "bold"), fg=C["accent"], bg=C["card"]).pack(anchor="w", pady=(0, 2))
        self.algo_var = tk.StringVar(value="Dyna-Q")
        algo_dropdown = ttk.Combobox(
            cfg_card, textvariable=self.algo_var, 
            values=["Q-Learning", "SARSA", "Exp SARSA", "D Q-Learning", "Dyna-Q", "Dyna-Q+"], 
            state="readonly", font=("Segoe UI", 9)
        )
        algo_dropdown.pack(fill="x", pady=(0, 6))
        algo_dropdown.bind("<<ComboboxSelected>>", self.on_algo_change)
        
        tk.Label(cfg_card, text="Hyperparameter Suite", font=("Segoe UI", 10, "bold"), fg=C["accent"], bg=C["card"]).pack(anchor="w", pady=(0, 2))
        self.slider_alpha = self.create_slider(cfg_card, "Learning Rate (α)", 0.01, 1.0, 0.1)
        self.slider_gamma = self.create_slider(cfg_card, "Discount Factor (γ)", 0.5, 0.99, 0.95)
        self.slider_epsilon = self.create_slider(cfg_card, "Exploration Rate (ε)", 0.0, 1.0, 0.1)
        self.slider_planning = self.create_slider(cfg_card, "Planning Steps (N)", 0, 50, 10, is_int=True)
        self.slider_kappa = self.create_slider(cfg_card, "Exploration Bonus (κ)", 0.0, 0.05, 0.005)

        btn_card = ttk.Frame(right_panel, style="Card.TFrame", padding=8)
        btn_card.pack(fill="x", pady=(0, 6))
        
        grid_btns = ttk.Frame(btn_card, style="Card.TFrame")
        grid_btns.pack(fill="x")
        grid_btns.columnconfigure(0, weight=1)
        grid_btns.columnconfigure(1, weight=1)
        
        ttk.Button(grid_btns, text=f"Train {TRAIN_EPISODES} Episodes", command=self.train_episodes_fast).grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        ttk.Button(grid_btns, text="Watch Learning Ep.", command=self.watch_episode_slow).grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        ttk.Button(grid_btns, text="Generate Maze Layout", command=self.generate_random_maze).grid(row=1, column=0, padx=2, pady=2, sticky="ew")
        ttk.Button(grid_btns, text="Play Best Route", command=self.play_fully_exploit).grid(row=1, column=1, padx=2, pady=2, sticky="ew")
        
        ttk.Button(btn_card, text="Clear Highlighted Method Stats", command=self.clear_stats).pack(fill="x", pady=(4, 2))
        ttk.Button(btn_card, text="Reset Engine Framework Completely", command=self.reset_all).pack(fill="x", pady=2)
        
        log_card = ttk.Frame(right_panel, style="Card.TFrame", padding=6)
        log_card.pack(fill="both", expand=True)
        self.log_area = tk.Text(log_card, height=4, bg=C["bg"], fg=C["text"], insertbackground=C["text"], font=("Consolas", 9), relief="solid", borderwidth=1)
        self.log_area.pack(fill="both", expand=True, pady=(2, 0))
        
        scrollbar = ttk.Scrollbar(self.log_area, command=self.log_area.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_area['yscrollcommand'] = scrollbar.set

    def create_slider(self, parent, label_text, min_v, max_v, init_v, is_int=False):
        frame = ttk.Frame(parent, style="Card.TFrame")
        frame.pack(fill="x", pady=1)
        lbl_val = tk.Label(frame, text=f"{label_text}: {init_v}", fg=C["text"], bg=C["card"], font=("Segoe UI", 9))
        lbl_val.pack(anchor="w")
        
        def on_slider_update(val):
            v = int(float(val)) if is_int else float(val)
            lbl_val.config(text=f"{label_text}: {v:.3f}" if not is_int else f"{label_text}: {v}")
            self.sync_agent_parameters()

        slider = ttk.Scale(frame, from_=min_v, to=max_v, value=init_v, command=on_slider_update)
        slider.pack(fill="x")
        return slider

    def log(self, message):
        self.log_area.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.log_area.see(tk.END)

    def sync_agent_parameters(self):
        self.agent.alpha = float(self.slider_alpha.get())
        self.agent.gamma = float(self.slider_gamma.get())
        self.agent.epsilon = float(self.slider_epsilon.get())
        self.agent.planning_steps = int(float(self.slider_planning.get()))
        self.agent.kappa = float(self.slider_kappa.get())
        self.agent.current_method = self.algo_var.get()

    def on_algo_change(self, event=None):
        self.agent.current_method = self.algo_var.get()
        self.log(f"Switched logic strategy profile to: {self.algo_var.get()}")
        self.draw_maze()
        self.draw_evaluation_graph()

    def canvas_paint_wall(self, event):
        r = event.y // CELL_SIZE
        c = event.x // CELL_SIZE
        if 0 <= r < ROWS and 0 <= c < COLS:
            if (r, c) != START and (r, c) != GOAL:
                self.env.blocks.add((r, c))
                self.draw_maze()

    def canvas_clear_wall(self, event):
        r = event.y // CELL_SIZE
        c = event.x // CELL_SIZE
        if 0 <= r < ROWS and 0 <= c < COLS:
            if (r, c) in self.env.blocks:
                self.env.blocks.remove((r, c))
                self.draw_maze()

    def draw_maze(self):
        self.canvas.delete("all")
        for r in range(ROWS):
            for c in range(COLS):
                x0, y0 = c * CELL_SIZE, r * CELL_SIZE
                x1, y1 = x0 + CELL_SIZE, y0 + CELL_SIZE
                
                if (r, c) in self.env.blocks:
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill=C["wall"], outline=C["grid_line"])
                elif (r, c) == GOAL:
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill=C["bg"], outline=C["grid_line"])
                    self.canvas.create_text(x0 + CELL_SIZE//2, y0 + CELL_SIZE//2, text="🧀", font=("Segoe UI", 16))
                elif (r, c) == START:
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill=C["bg"], outline=C["grid_line"])
                    self.canvas.create_text(x0 + CELL_SIZE//2, y0 + CELL_SIZE//2, text="🏁", font=("Segoe UI", 14))
                else:
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill=C["card"], outline=C["grid_line"])

                if (r, c) not in self.env.blocks and (r, c) != GOAL:
                    q_vals = [self.agent.get_q((r, c), a) for a in ACTIONS]
                    max_q = max(q_vals)
                    if max_q > 1e-4:
                        best_acts = [a for a, q in zip(ACTIONS, q_vals) if abs(q - max_q) < 1e-4]
                        self.draw_policy_arrows(x0, y0, best_acts)

        mr, mc = self.env.state
        self.mouse_id = self.canvas.create_text(mc * CELL_SIZE + CELL_SIZE//2, mr * CELL_SIZE + CELL_SIZE//2, text="🐭", font=("Segoe UI", 18))

    def draw_policy_arrows(self, x0, y0, best_actions):
        pad = 8
        cx, cy = x0 + CELL_SIZE // 2, y0 + CELL_SIZE // 2
        color = self.algo_colors.get(self.algo_var.get(), C["accent"])
        
        for action in best_actions:
            if action == "up": self.canvas.create_line(cx, cy, cx, y0 + pad, arrow=tk.LAST, fill=color, width=1)
            elif action == "down": self.canvas.create_line(cx, cy, cx, y0 + CELL_SIZE - pad, arrow=tk.LAST, fill=color, width=1)
            elif action == "left": self.canvas.create_line(cx, cy, x0 + pad, cy, arrow=tk.LAST, fill=color, width=1)
            elif action == "right": self.canvas.create_line(cx, cy, x0 + CELL_SIZE - pad, cy, arrow=tk.LAST, fill=color, width=1)

    def update_mouse_visual(self):
        mr, mc = self.anim_state
        self.canvas.coords(self.mouse_id, mc * CELL_SIZE + CELL_SIZE//2, mr * CELL_SIZE + CELL_SIZE//2)
        self.root.update_idletasks()

    def generate_random_maze(self):
        self.env.blocks.clear()
        protected = {START, GOAL, (0, 7), (1, 8), (3, 3), (3, 5), (4, 4)}
        for r in range(ROWS):
            for c in range(COLS):
                if (r, c) not in protected and random.random() < 0.22:
                    self.env.blocks.add((r, c))
        self.log("Random block barriers applied successfully.")
        self.draw_maze()

    def train_episodes_fast(self):
        if self.is_running: return
        self.is_running = True
        self.sync_agent_parameters()
        algo = self.algo_var.get()
        self.log(f"Processing {TRAIN_EPISODES} training cycles using {algo} logic framework...")
        
        try:
            steps_history = []
            for ep in range(TRAIN_EPISODES):
                state = self.env.reset()
                done = False
                steps = 0
                action = self.agent.choose_action(state) if algo == "SARSA" else None

                while not done and steps < 500:
                    if algo == "SARSA":
                        next_state, reward, done = self.env.step(action)
                        next_action = self.agent.choose_action(next_state)
                        self.agent.learn_transition(state, action, reward, next_state, next_action, method=algo)
                        state, action = next_state, next_action
                    else:
                        act = self.agent.choose_action(state)
                        next_state, reward, done = self.env.step(act)
                        self.agent.learn_transition(state, act, reward, next_state, method=algo)
                        state = next_state
                    steps += 1
                steps_history.append(steps)
                self.eval_history[algo].append(steps)
            self.log(f"Cycle completed. Last episode path duration: {steps_history[-1]} steps.")
        except Exception as e:
            self.log(f"Error encountered during fast train pipeline: {str(e)}")
        
        self.is_running = False
        self.env.reset()
        self.draw_maze()
        self.draw_evaluation_graph()

    def watch_episode_slow(self):
        if self.is_running: return
        self.is_running = True
        self.sync_agent_parameters()
        algo = self.algo_var.get()
        
        self.anim_state = self.env.reset()
        self.draw_maze()
        self.log(f"Rendering frame loop for active {algo} learning run...")
        self.anim_action = self.agent.choose_action(self.anim_state) if algo == "SARSA" else None
        self.anim_steps = 0
        self._watch_loop(algo)

    def _watch_loop(self, algo):
        if not self.is_running: return
        
        if self.env.done or self.anim_steps >= 300:
            self.log(f"Visual loop completed in {self.anim_steps} frames.")
            self.eval_history[algo].append(self.anim_steps)
            self.draw_evaluation_graph()
            self.is_running = False
            self.env.reset()
            self.draw_maze()
            return
        
        try:
            if algo == "SARSA":
                next_state, reward, done = self.env.step(self.anim_action)
                next_action = self.agent.choose_action(next_state)
                self.agent.learn_transition(self.anim_state, self.anim_action, reward, next_state, next_action, method=algo)
                self.anim_state = next_state
                self.anim_action = next_action
            else:
                action = self.agent.choose_action(self.anim_state)
                next_state, reward, done = self.env.step(action)
                self.agent.learn_transition(self.anim_state, action, reward, next_state, method=algo)
                self.anim_state = next_state

            self.anim_steps += 1
            self.update_mouse_visual()
            self.root.after(80, lambda: self._watch_loop(algo))
        except Exception as e:
            self.log(f"Animation execution fault: {str(e)}")
            self.is_running = False

    def play_fully_exploit(self):
        if self.is_running: return
        if not self.agent.Q and not self.agent.Q1:
            self.log("No value map policies exist. Run evaluation batches first.")
            return

        self.is_running = True
        self.anim_state = self.env.reset()
        self.draw_maze()
        self.log(f"Projecting pure deterministic exploitation route...")
        self.anim_steps = 0
        self._play_loop()

    def _play_loop(self):
        if not self.is_running: return
        
        if self.env.done or self.anim_steps >= 100:
            if self.env.done: self.log(f"Optimal trajectory targeted goal successfully in {self.anim_steps} steps.")
            else: self.log("Exploitation timeout: Pathing logic trapped.")
            self.is_running = False
            self.env.reset()
            self.draw_maze()
            return
        
        action = self.agent.choose_action(self.anim_state, exploit_only=True)
        self.anim_state, _, _ = self.env.step(action)
        self.anim_steps += 1
        self.update_mouse_visual()
        self.root.after(100, self._play_loop)

    def clear_stats(self):
        algo = self.algo_var.get()
        self.eval_history[algo].clear()
        self.log(f"Metrics vector dropped for {algo}.")
        self.draw_evaluation_graph()

    def reset_all(self):
        self.is_running = False
        self.agent.reset_knowledge()
        for algo in self.eval_history: self.eval_history[algo].clear()
        self.env.reset()
        self.draw_maze()
        self.draw_evaluation_graph()
        self.log("Global matrices, knowledge bases, and vectors flushed.")

    def draw_evaluation_graph(self):
        self.graph_canvas.delete("all")
        w = self.graph_canvas.winfo_width()
        h = self.graph_canvas.winfo_height()
        if w < 10: w, h = 580, 115
            
        pad_l, pad_r, pad_t, pad_b = 35, 130, 10, 20
        graph_w, graph_h = w - pad_l - pad_r, h - pad_t - pad_b
        
        self.graph_canvas.create_rectangle(pad_l, pad_t, pad_l + graph_w, pad_t + graph_h, fill=C["card"], outline=C["grid_line"])
        
        ticks = [1, 5, 20, 100, 500]
        for tick in ticks:
            y_ratio = np.log10(tick) / np.log10(500)
            y = pad_t + graph_h - (y_ratio * graph_h)
            self.graph_canvas.create_line(pad_l, y, pad_l + graph_w, y, fill=C["grid_line"], dash=(1, 2))
            self.graph_canvas.create_text(pad_l - 12, y, text=str(tick), fill=C["text"], font=("Segoe UI", 7))

        max_len = max([len(self.eval_history[k]) for k in self.eval_history] + [1])
        x_ticks = min(8, max_len)
        for i in range(x_ticks):
            idx = int((i / max(1, x_ticks - 1)) * (max_len - 1))
            x = pad_l + (idx / max(1, max_len - 1)) * graph_w
            self.graph_canvas.create_line(x, pad_t, x, pad_t + graph_h, fill=C["grid_line"], dash=(1, 2))
            self.graph_canvas.create_text(x, pad_t + graph_h + 8, text=f"Ep.{idx+1}", fill=C["text"], font=("Segoe UI", 7))

        for algo, history in self.eval_history.items():
            if len(history) < 2: continue
            points = []
            for idx, steps in enumerate(history):
                x = pad_l + (idx / (max_len - 1)) * graph_w
                y_ratio = np.log10(max(1, steps)) / np.log10(500)
                y = pad_t + graph_h - (y_ratio * graph_h)
                points.append((x, y))
                
            for i in range(len(points) - 1):
                self.graph_canvas.create_line(points[i][0], points[i][1], points[i+1][0], points[i+1][1], fill=self.algo_colors[algo], width=1.5)
                
        legend_x = pad_l + graph_w + 10
        for i, (algo, color) in enumerate(self.algo_colors.items()):
            ly = pad_t + 2 + (i * 15)
            self.graph_canvas.create_rectangle(legend_x, ly, legend_x + 10, ly + 10, fill=color, outline="")
            
            hist_len = len(self.eval_history[algo])
            last_val = f" ({self.eval_history[algo][-1]}s)" if hist_len > 0 else " (-)"
            fw = "bold" if algo == self.algo_var.get() else "normal"
            tc = C["text"] if algo == self.algo_var.get() else C["wall"]
            
            self.graph_canvas.create_text(legend_x + 15, ly + 5, text=f"{algo}{last_val}", fill=tc, font=("Segoe UI", 8, fw), anchor="w")

    def run(self):
        self.root.update()
        self.graph_canvas.bind("<Configure>", lambda e: self.draw_evaluation_graph())
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = EducationalSuite(root)
    app.run()