"""
Actor-Critic Calculation Demo
For AI Engineering (Fundamental) Class, LU Lab., Myanmar
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import random

# --- CONFIG ---
ROWS, COLS = 5, 5
CELL_SIZE = 70 
OFFSET = 25  # Space for X/Y axis labels
GOAL = (0, 4)
TRAPS = [(1, 3), (2, 3), (3, 3)]
WALLS = [(2, 1), (2, 2)]

class TDEngine:
    def __init__(self):
        self.V = { (r, c): 0.0 for r in range(ROWS) for c in range(COLS) }
        self.gamma = 0.9
        self.alpha = 0.1  
        self.episode_history = [] 
        self.episode_count = 0

    def generate_episode(self):
        state = (4, 0)
        path = []
        while state not in [GOAL] + TRAPS:
            action = random.choice(["up", "down", "left", "right"])
            r, c = state
            if action == "up": r = max(0, r-1)
            elif action == "down": r = min(ROWS-1, r+1)
            elif action == "left": c = max(0, c-1)
            elif action == "right": c = min(COLS-1, c+1)
            
            next_state = (r, c) if (r, c) not in WALLS else state
            reward = 1.0 if next_state == GOAL else (-1.0 if next_state in TRAPS else -0.04)
            
            path.append((state, action, reward, next_state))
            state = next_state
            if len(path) > 60: break 
        
        self.episode_history = path
        self.episode_count += 1
        return path

    def get_detailed_log(self):
        log = f"--- EPISODE {self.episode_count} CALCULATION (TD(0)) ---\n"
        log += "Notice: We update V(S) FORWARD, step-by-step, using V(S')!\n\n"
        
        for t, (state, action, reward, next_s) in enumerate(self.episode_history):
            next_v = 0.0 if next_s in [GOAL] + TRAPS else self.V[next_s]
            old_v = self.V[state]
            delta = reward + (self.gamma * next_v) - old_v
            self.V[state] = old_v + (self.alpha * delta)
            
            log += f"Step {t+1}: S{state} -> {action} -> S{next_s} | R: {reward:+.2f}\n"
            log += f"  Target   = R + γ*V(S') = {reward:+.2f} + ({self.gamma} * {next_v:.2f}) = {reward + self.gamma * next_v:.3f}\n"
            log += f"  TD Error = Target - V(S) = {reward + self.gamma * next_v:.3f} - {old_v:.2f} = {delta:+.3f}\n"
            log += f"  Update   = V(S) + α*δ = {old_v:.2f} + ({self.alpha} * {delta:+.3f}) = {self.V[state]:.4f}\n\n"
            
        return log

class TDVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Temporal Difference Learning Lab (TD(0))")
        self.root.geometry("1100x700")
        self.engine = TDEngine()
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
        ttk.Button(ctrl, text="2. Calculate & Learn (Step-by-Step)", command=self.run_learn).pack(fill="x", pady=2)
        ttk.Button(ctrl, text="Toggle Policy Arrows", command=self.toggle_policy).pack(fill="x", pady=2)
        self.btn_reset = ttk.Button(ctrl, text="3. Reset Memory", command=self.reset_learning)
        self.btn_reset.pack(fill="x", pady=2)
        
        right_panel = ttk.Frame(paned)
        paned.add(right_panel, weight=2)
        
        theory_frame = ttk.LabelFrame(right_panel, text="Theory & Guide: Compare with MC!")
        theory_frame.pack(fill="x", pady=5)
        
        theory_text = (
            "MATH: V(S) ← V(S) + α * [R + γ*V(S') - V(S)]\n"
            "   - δ (TD Error): The surprise. How much better/worse than expected?\n"
            "   - R + γ*V(S'): The new Target (Bootstrapped guess).\n\n"
            "MC vs TD:\n"
            "   - MC waits for episode end, uses actual Return (G).\n"
            "   - TD updates EVERY step, uses a guess (V(S')).\n\n"
            "COLORS:\n"
            "   - Green: High Value   - Red: Low Value"
        )
        ttk.Label(theory_frame, text=theory_text, font=("Consolas", 10), justify="left", padding=10).pack(anchor="w")
        
        log_frame = ttk.LabelFrame(right_panel, text="Calculation Log")
        log_frame.pack(fill="both", expand=True)
        self.console = scrolledtext.ScrolledText(log_frame, font=("Consolas", 9), bg="#fdfdfd")
        self.console.pack(fill="both", expand=True)
        
        self.draw_grid()

    def toggle_policy(self):
        self.show_policy = not self.show_policy
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
        path_map = {}
        for i, step in enumerate(path):
            state = step[0]
            if state not in path_map:
                path_map[state] = i + 1 
        
        start_state = path[0][0] if path else None
        end_state = path[-1][0] if path else None

        for r in range(ROWS):
            for c in range(COLS):
                state = (r, c)
                x, y = c * CELL_SIZE + OFFSET, r * CELL_SIZE + OFFSET
                
                val = self.engine.V[state]
                bg_color = "white"
                if state in WALLS: bg_color = "#333"
                elif state == GOAL: bg_color = "#d1e7dd"
                elif state in TRAPS: bg_color = "#f8d7da"
                elif val > 0: bg_color = f"#{int(255-min(val,1)*150):02x}ff{int(255-min(val,1)*150):02x}"
                elif val < 0: bg_color = f"#ff{int(255-abs(val)*100):02x}{int(255-abs(val)*100):02x}"
                
                self.canvas.create_rectangle(x, y, x+CELL_SIZE, y+CELL_SIZE, fill=bg_color, outline="#eee")
                
                if state in path_map:
                    self.canvas.create_oval(x+5, y+5, x+25, y+25, fill="#444", outline="white")
                    self.canvas.create_text(x+15, y+15, text=str(path_map[state]), font=("Arial", 9, "bold"), fill="white")
                    
                    if state == start_state:
                        self.canvas.create_text(x+45, y+12, text="[S]", font=("Arial", 9, "bold"), fill="blue")
                    if state == end_state:
                        self.canvas.create_text(x+45, y+12, text="[E]", font=("Arial", 9, "bold"), fill="red")

                self.canvas.create_text(x+35, y+40, text=f"{val:.2f}", font=("Arial", 9, "bold"))
                
                if self.show_policy and state not in WALLS and state not in [GOAL] + TRAPS:
                    self.draw_arrow(x+35, y+55, state)

    def draw_arrow(self, cx, cy, state):
        neighbors = {"up": (state[0]-1, state[1]), "down": (state[0]+1, state[1]), "left": (state[0], state[1]-1), "right": (state[0], state[1]+1)}
        best_val = -float('inf'); best_dir = None
        for d, pos in neighbors.items():
            if 0 <= pos[0] < ROWS and 0 <= pos[1] < COLS and pos not in WALLS:
                if self.engine.V[pos] > best_val: best_val = self.engine.V[pos]; best_dir = d
        if best_dir:
            arr = {"up": (0, -10), "down": (0, 10), "left": (-10, 0), "right": (10, 0)}
            dx, dy = arr[best_dir]
            self.canvas.create_line(cx, cy, cx+dx, cy+dy, arrow=tk.LAST, fill="black", width=2)

    def run_episode(self):
        path = self.engine.generate_episode()
        self.draw_grid()
        self.console.insert(tk.END, f"\n--- New Episode Path (Len: {len(path)}) ---\n")

    def run_learn(self):
        self.console.insert(tk.END, self.engine.get_detailed_log() + "\n")
        self.draw_grid()
    
    def reset_learning(self):
        self.engine.V = { (r, c): 0.0 for r in range(ROWS) for c in range(COLS) }
        self.engine.episode_history = []
        self.console.delete("1.0", tk.END)
        self.draw_grid()

if __name__ == "__main__":
    root = tk.Tk()
    app = TDVisualizer(root)
    root.mainloop()
 