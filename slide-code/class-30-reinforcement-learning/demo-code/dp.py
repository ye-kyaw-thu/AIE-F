"""
Actor-Critic Calculation Demo
For AI Engineering (Fundamental) Class, LU Lab., Myanmar
"""

import tkinter as tk
from tkinter import ttk
import sys

# --- THEME ENGINE ---
THEMES = {
    "light": {"bg": "#f8f9fa", "card": "#ffffff", "text": "#212529", "accent": "#0d6efd", "success": "#198754", "warn": "#dc3545", "wall": "#343a40", "grid_line": "#dee2e6", "highlight": "#fff3cd"},
    "dark": {"bg": "#121212", "card": "#1e1e1e", "text": "#e0e0e0", "accent": "#3b82f6", "success": "#10b981", "warn": "#ef4444", "wall": "#374151", "grid_line": "#333333", "highlight": "#453208"},
    "oceanic": {"bg": "#0f172a", "card": "#1e293b", "text": "#f8fafc", "accent": "#38bdf8", "success": "#34d399", "warn": "#f87171", "wall": "#475569", "grid_line": "#334155", "highlight": "#78350f"}
}

SELECTED_THEME = "light"
for arg in sys.argv:
    if arg.startswith("--theme="):
        t_val = arg.split("=")[1].lower()
        if t_val in THEMES: SELECTED_THEME = t_val

C = THEMES[SELECTED_THEME]

# --- ENVIRONMENT CONFIGURATION ---
ROWS = 5
COLS = 5
CELL_SIZE = 80
OFFSET = 25  # Space for X/Y axis labels
GOAL = (0, 4)
TRAPS = [(1, 3), (2, 3), (3, 3)]
WALLS = [(2, 1), (2, 2)]
ACTIONS = ["up", "down", "left", "right"]

class GridMDP:
    def __init__(self, slip_prob=0.0):
        self.slip_prob = slip_prob
        self.gamma = 0.9

    def get_transitions(self, state, action):
        if state == GOAL:
            return [(1.0, state, 0.0, True)]
        if state in TRAPS:
            return [(1.0, state, 0.0, True)]
            
        transitions = []
        intended_prob = 1.0 - self.slip_prob
        slip_prob_each = self.slip_prob / 2.0
        
        action_probs = {}
        if action == "up": action_probs = {"up": intended_prob, "left": slip_prob_each, "right": slip_prob_each}
        elif action == "down": action_probs = {"down": intended_prob, "left": slip_prob_each, "right": slip_prob_each}
        elif action == "left": action_probs = {"left": intended_prob, "up": slip_prob_each, "down": slip_prob_each}
        elif action == "right": action_probs = {"right": intended_prob, "up": slip_prob_each, "down": slip_prob_each}

        for act, prob in action_probs.items():
            if prob == 0: continue
            r, c = state
            if act == "up": r -= 1
            elif act == "down": r += 1
            elif act == "left": c -= 1
            elif act == "right": c += 1
            
            if r < 0 or r >= ROWS or c < 0 or c >= COLS or (r, c) in WALLS:
                next_state = state
            else:
                next_state = (r, c)
                
            if next_state == GOAL: reward = 1.0
            elif next_state in TRAPS: reward = -1.0
            else: reward = -0.04 
            
            is_terminal = next_state == GOAL or next_state in TRAPS
            transitions.append((prob, next_state, reward, is_terminal))
            
        return transitions

class DynamicProgrammingCore:
    def __init__(self, mdp):
        self.mdp = mdp
        self.V = { (r, c): 0.0 for r in range(ROWS) for c in range(COLS) }
        self.pi = { (r, c): {a: 0.25 for a in ACTIONS} for r in range(ROWS) for c in range(COLS) }

    def reset(self):
        self.V = { (r, c): 0.0 for r in range(ROWS) for c in range(COLS) }
        self.pi = { (r, c): {a: 0.25 for a in ACTIONS} for r in range(ROWS) for c in range(COLS) }

class DPSuiteGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Dynamic Programming Inspector [Theme: {SELECTED_THEME.title()}]")
        self.root.configure(bg=C["bg"])
        
        self.mdp = GridMDP()
        self.dp = DynamicProgrammingCore(self.mdp)
        self.focused_cell = None
        self.show_policy = False  
        
        self.build_ui()
        self.draw_grid()

    def build_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', background=C["bg"], foreground=C["text"])
        style.configure('Card.TFrame', background=C["card"], borderwidth=1, relief="solid", bordercolor=C["grid_line"])
        
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill="both", expand=True)
        
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        tk.Label(left_panel, text="Step 1: Click any cell to calculate its value manually.", font=("Segoe UI", 12, "bold"), fg=C["accent"], bg=C["bg"]).pack(anchor="w", pady=(0, 5))
        
        self.canvas = tk.Canvas(left_panel, width=COLS*CELL_SIZE + OFFSET, height=ROWS*CELL_SIZE + OFFSET, bg=C["card"], highlightthickness=1, highlightbackground=C["grid_line"])
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_cell_click)
        
        env_frame = ttk.Frame(left_panel, padding=5)
        env_frame.pack(fill="x", pady=10)
        
        self.gamma_lbl = tk.Label(env_frame, text=f"Discount (γ): {self.mdp.gamma}", bg=C["bg"], fg=C["text"])
        self.gamma_lbl.grid(row=0, column=0, sticky="w")
        self.gamma_scale = ttk.Scale(env_frame, from_=0.1, to=1.0, value=self.mdp.gamma, command=self.update_gamma)
        self.gamma_scale.grid(row=0, column=1, sticky="ew", padx=10)
        
        self.slip_lbl = tk.Label(env_frame, text=f"Slip Prob: {self.mdp.slip_prob}", bg=C["bg"], fg=C["text"])
        self.slip_lbl.grid(row=1, column=0, sticky="w")
        self.slip_scale = ttk.Scale(env_frame, from_=0.0, to=0.8, value=self.mdp.slip_prob, command=self.update_slip)
        self.slip_scale.grid(row=1, column=1, sticky="ew", padx=10)
        
        ttk.Button(env_frame, text="Reset Engine", command=self.reset_all).grid(row=0, column=2, rowspan=2, padx=10)

        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(fill="x", pady=(0, 5))
        ttk.Button(btn_frame, text="Toggle Policy Arrows", command=self.toggle_policy).pack(fill="x")

        right_panel = ttk.Frame(main_frame, style="Card.TFrame", padding=10)
        right_panel.pack(side="right", fill="both", expand=True)
        
        tk.Label(right_panel, text="Bellman Math Inspector", font=("Segoe UI", 12, "bold"), fg=C["success"], bg=C["card"]).pack(anchor="w", pady=(0, 5))
        tk.Label(right_panel, text="Formula: Q(s,a) = Σ p(s'|s,a) * [R + γ * V(s')]", font=("Consolas", 9), fg=C["wall"], bg=C["card"]).pack(anchor="w", pady=(0, 5))
        
        self.math_console = tk.Text(right_panel, width=55, height=25, bg="#1e1e1e" if SELECTED_THEME == "light" else "#000000", fg="#00ff00", font=("Consolas", 9), relief="flat")
        self.math_console.pack(fill="both", expand=True)
        
        self.write_console("System initialized. Awaiting manual cell inspection...")

    def toggle_policy(self):
        self.show_policy = not self.show_policy
        self.draw_grid()

    def update_gamma(self, val):
        self.mdp.gamma = round(float(val), 2)
        self.gamma_lbl.config(text=f"Discount (γ): {self.mdp.gamma}")

    def update_slip(self, val):
        self.mdp.slip_prob = round(float(val), 2)
        self.slip_lbl.config(text=f"Slip Prob: {self.mdp.slip_prob}")

    def write_console(self, text):
        self.math_console.delete(1.0, tk.END)
        self.math_console.insert(tk.END, text)

    def on_cell_click(self, event):
        c = (event.x - OFFSET) // CELL_SIZE
        r = (event.y - OFFSET) // CELL_SIZE
        state = (r, c)
        
        if r < 0 or r >= ROWS or c < 0 or c >= COLS: return
        if state in WALLS or state == GOAL or state in TRAPS:
            self.write_console(f"State {state} is terminal or impassable.\nNo standard Bellman update required.")
            self.focused_cell = state
            self.draw_grid()
            return
            
        self.focused_cell = state
        self.calculate_and_inspect(state)

    def calculate_and_inspect(self, state):
        log = f"--- EVALUATING STATE {state} ---\n"
        log += f"Discount (γ) = {self.mdp.gamma} | Slip = {self.mdp.slip_prob}\n\n"
        
        q_values = {}
        for action in ACTIONS:
            log += f"Action: [{action.upper()}]\n"
            q_val = 0.0
            transitions = self.mdp.get_transitions(state, action)
            
            for prob, nxt_s, reward, is_term in transitions:
                term_str = " (Terminal)" if is_term else ""
                v_nxt = 0.0 if is_term else self.dp.V[nxt_s]
                
                step_val = prob * (reward + self.mdp.gamma * v_nxt)
                q_val += step_val
                
                log += f"  -> Prob {prob:.2f} lands on {nxt_s}{term_str}\n"
                log += f"     Formula: {prob:.2f} * ({reward:+.2f} + {self.mdp.gamma} * {v_nxt:.2f})\n"
                log += f"     Value  : {step_val:+.4f}\n"
                
            q_values[action] = q_val
            log += f"  >> Q({state}, {action}) = {q_val:+.4f}\n\n"
            
        max_q = max(q_values.values())
        best_acts = [a for a, q in q_values.items() if abs(q - max_q) < 1e-5]
        
        old_v = self.dp.V[state]
        self.dp.V[state] = max_q
        
        for a in ACTIONS:
            self.dp.pi[state][a] = 1.0 / len(best_acts) if a in best_acts else 0.0
            
        log += "--- UPDATE SUMMARY ---\n"
        log += f"Best Action(s) : {', '.join(best_acts).upper()}\n"
        log += f"Old V(s)       : {old_v:+.4f}\n"
        log += f"New V(s)       : {self.dp.V[state]:+.4f}\n"
        
        self.write_console(log)
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
        
        for r in range(ROWS):
            for c in range(COLS):
                state = (r, c)
                x0, y0 = c * CELL_SIZE + OFFSET, r * CELL_SIZE + OFFSET
                x1, y1 = x0 + CELL_SIZE, y0 + CELL_SIZE
                
                fill_color = C["bg"]
                if state in WALLS: fill_color = C["wall"]
                elif state == GOAL: fill_color = C["success"]
                elif state in TRAPS: fill_color = C["warn"]
                elif state == self.focused_cell: fill_color = C["highlight"]
                
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill_color, outline=C["grid_line"])
                
                if state in WALLS: continue
                
                if state == GOAL:
                    self.canvas.create_text(x0 + CELL_SIZE/2, y0 + CELL_SIZE/2, text="GOAL\n+1.0", fill="#fff", font=("Arial", 10, "bold"), justify="center")
                    continue
                if state in TRAPS:
                    self.canvas.create_text(x0 + CELL_SIZE/2, y0 + CELL_SIZE/2, text="TRAP\n-1.0", fill="#fff", font=("Arial", 10, "bold"), justify="center")
                    continue
                    
                v_str = f"{self.dp.V[state]:.2f}"
                text_color = C["text"] if state != self.focused_cell else "#000"
                self.canvas.create_text(x0 + CELL_SIZE/2, y0 + CELL_SIZE/2, text=v_str, fill=text_color, font=("Arial", 11, "bold"))
                
                if self.show_policy and state not in WALLS and state != GOAL and state not in TRAPS:
                    best_prob = max(self.dp.pi[state].values())
                    if best_prob > 0:
                        best_acts = [a for a in ACTIONS if self.dp.pi[state][a] == best_prob]
                        arr = {"up": (0, -25), "down": (0, 25), "left": (-25, 0), "right": (25, 0)}
                        cx, cy = x0 + CELL_SIZE/2, y0 + CELL_SIZE/2
                        for best_act in best_acts:
                            dx, dy = arr[best_act]
                            self.canvas.create_line(cx + (dx * 0.4), cy + (dy * 0.4), 
                                                     cx + dx, cy + dy, 
                                                     arrow=tk.LAST, width=2, fill=C["accent"])

    def reset_all(self):
        self.focused_cell = None
        self.dp.reset()
        self.write_console("Engine reset. Click a cell to begin manual value iteration.")
        self.draw_grid()

if __name__ == "__main__":
    root = tk.Tk()
    app = DPSuiteGUI(root)
    root.mainloop()
