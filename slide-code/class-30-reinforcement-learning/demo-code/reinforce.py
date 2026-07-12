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
CELL_SIZE = 70 
OFFSET = 25  # Space for X/Y axis labels
GOAL = (0, 4)
TRAPS = [(1, 3), (2, 3), (3, 3)]
WALLS = [(2, 1), (2, 2)]

class REINFORCEEngine:
    def __init__(self):
        self.prefs = { (r, c): {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0} 
                       for r in range(ROWS) for c in range(COLS) }
        self.alpha = 0.5 
        self.gamma = 0.9 
        self.episode_history = [] 
        self.episode_count = 0

    def get_probs(self, state):
        p = self.prefs[state]
        exps = {a: math.exp(val) for a, val in p.items()}
        sum_exps = sum(exps.values())
        return {a: e / sum_exps for a, e in exps.items()}
        
    def generate_episode(self):
        state = (4, 0)
        path = []
        epsilon = 0.2  
        
        for _ in range(60): 
            if state in [GOAL] + TRAPS: break
            
            probs = self.get_probs(state)
            
            if random.random() < epsilon:
                action = random.choice(["up", "down", "left", "right"])
            else:
                action = random.choices(list(probs.keys()), weights=list(probs.values()))[0]
            
            r, c = state
            if action == "up": r = max(0, r-1)
            elif action == "down": r = min(ROWS-1, r+1)
            elif action == "left": c = max(0, c-1)
            elif action == "right": c = min(COLS-1, c+1)
            
            next_state = (r, c) if (r, c) not in WALLS else state
            reward = 1.0 if next_state == GOAL else (-1.0 if next_state in TRAPS else -0.04)
            path.append((state, action, reward))
            state = next_state
        self.episode_history = path
        self.episode_count += 1
        return path

    def get_detailed_log(self):
        G = 0
        log = f"--- EPISODE {self.episode_count} CALCULATION ---\n"
        for t, (state, action, reward) in enumerate(reversed(self.episode_history)):
            G = reward + (self.gamma * G)
            probs = self.get_probs(state)
            old_pref = self.prefs[state][action]
            indicator = 1.0
            grad = indicator - probs[action]
            update = self.alpha * G * grad
            self.prefs[state][action] += update
            
            log += f"Step {t+1}: S{state} | Act: {action} | G: {G:.2f}\n"
            log += f"  Pref {action}: {old_pref:.2f} -> {self.prefs[state][action]:.2f} (Grad: {grad:.2f})\n"
        return log

class PolicyVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Policy Gradient Lab (REINFORCE)")
        self.root.geometry("1100x700")
        self.engine = REINFORCEEngine()
        self.show_policy = False
        
        paned = ttk.PanedWindow(root, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=5, pady=5)
        
        left_panel = ttk.Frame(paned)
        paned.add(left_panel, weight=1)
        self.canvas = tk.Canvas(left_panel, width=COLS*CELL_SIZE + OFFSET, height=ROWS*CELL_SIZE + OFFSET, bg="white")
        self.canvas.pack(pady=10)
        
        ctrl = ttk.Frame(left_panel)
        ctrl.pack(fill="x")
        ttk.Button(ctrl, text="1. Run Episode", command=self.run_episode).pack(fill="x", pady=2)
        ttk.Button(ctrl, text="2. Calculate & Update", command=self.run_learn).pack(fill="x", pady=2)
        ttk.Button(ctrl, text="Toggle Policy Arrows", command=self.toggle_policy).pack(fill="x", pady=2)
        ttk.Button(ctrl, text="3. Reset Memory", command=self.reset_learning).pack(fill="x", pady=2)
        
        right_panel = ttk.Frame(paned)
        paned.add(right_panel, weight=2)
        
        theory_frame = ttk.LabelFrame(right_panel, text="Theory & Guide")
        theory_frame.pack(fill="x", pady=5)
        theory_text = (
            "MATH: θ ← θ + α * G * ∇log π(a|s)\n"
            "   - G: Return (Discounted future rewards)\n"
            "   - α: Learning Rate\n"
            "   - ∇log π: Policy Gradient\n\n"
            "COLORS:\n"
            "   - Heatmap: Agent's Preference (Logit) Intensity"
        )
        ttk.Label(theory_frame, text=theory_text, font=("Consolas", 10), justify="left", padding=10).pack(anchor="w")
        
        self.console = scrolledtext.ScrolledText(right_panel, font=("Consolas", 9), bg="#fdfdfd")
        self.console.pack(fill="both", expand=True)
        
        self.draw_grid()

    def toggle_policy(self):
        self.show_policy = not self.show_policy
        self.draw_grid()

    def run_episode(self):
        self.engine.generate_episode()
        self.draw_grid()
        self.console.insert(tk.END, f"\n--- Episode {self.engine.episode_count} Finished ---\n")

    def run_learn(self):
        self.console.insert(tk.END, self.engine.get_detailed_log() + "\n")
        self.draw_grid()

    def reset_learning(self):
        self.engine.prefs = { (r, c): {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0} 
                              for r in range(ROWS) for c in range(COLS) }
        self.engine.episode_history = []
        self.console.delete("1.0", tk.END)
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
                
                val = max(self.engine.prefs[state].values())
                bg = "white"
                if state in WALLS: bg = "#333"
                elif state == GOAL: bg = "#d1e7dd"
                elif state in TRAPS: bg = "#f8d7da"
                elif val > 0: 
                    intensity = max(0, 255 - int(val * 50))
                    bg = f"#{intensity:02x}ff{intensity:02x}"
                elif val < 0: 
                    intensity = max(0, 255 - int(abs(val) * 50))
                    bg = f"#ff{intensity:02x}{intensity:02x}"
                
                self.canvas.create_rectangle(x, y, x+CELL_SIZE, y+CELL_SIZE, fill=bg, outline="#eee")
                
                if state not in WALLS and state not in [GOAL] + TRAPS:
                    probs = self.engine.get_probs(state)
                    text_pos = {"up": (x+CELL_SIZE/2, y+15), "down": (x+CELL_SIZE/2, y+CELL_SIZE-15), 
                                "left": (x+15, y+CELL_SIZE/2), "right": (x+CELL_SIZE-15, y+CELL_SIZE/2)}
                    for act, p in probs.items():
                        tx, ty = text_pos[act]
                        self.canvas.create_text(tx, ty, text=f"{p:.2f}", font=("Arial", 7), fill="blue")

                if state in path_map:
                    self.canvas.create_oval(x+5, y+5, x+25, y+25, fill="#444", outline="white")
                    self.canvas.create_text(x+15, y+15, text=str(path_map[state]), font=("Arial", 9, "bold"), fill="white")
                    if state == start_state: self.canvas.create_text(x+CELL_SIZE-15, y+15, text="[S]", font=("Arial", 9, "bold"), fill="blue")
                    if state == end_state: self.canvas.create_text(x+CELL_SIZE-15, y+15, text="[E]", font=("Arial", 9, "bold"), fill="red")

                if self.show_policy and state not in WALLS and state not in [GOAL] + TRAPS:
                    probs = self.engine.get_probs(state)
                    best_act = max(probs, key=probs.get)
                    arr = {"up": (0, -20), "down": (0, 20), "left": (-20, 0), "right": (20, 0)}
                    dx, dy = arr[best_act]
                    cx, cy = x + CELL_SIZE/2, y + CELL_SIZE/2
                    self.canvas.create_line(cx, cy, cx+dx, cy+dy, arrow=tk.LAST, width=2)

if __name__ == "__main__":
    root = tk.Tk()
    app = PolicyVisualizer(root)
    root.mainloop()
