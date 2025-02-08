import tkinter as tk
from tkinter import ttk, messagebox, font

class ForwardPassVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Forward Pass Visualizer")
        
        # Configure window
        self.root.geometry("1000x1200")  # Set fixed window size
        
        # Configure larger custom fonts
        self.header_font = font.Font(size=18, weight='bold')
        self.input_font = font.Font(size=16)
        self.result_font = font.Font(size=24)
        self.step_font = font.Font(size=20)
        self.button_font = font.Font(size=16)
        
        # Configure grid weights for centering
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        self.current_step = 0
        self.setup_gui()
        
    def setup_gui(self):
        # Main container frame
        container = ttk.Frame(self.root)
        container.grid(row=0, column=0, sticky="nsew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        
        # Center frame for all content
        center_frame = ttk.Frame(container, padding="20")
        center_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights for center frame
        for i in range(4):  # Adjust based on number of rows
            center_frame.grid_rowconfigure(i, weight=1)
        center_frame.grid_columnconfigure(0, weight=1)
        
        # Input matrices frame
        input_frame = ttk.LabelFrame(center_frame, text="Input Matrices", padding="20")
        input_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 20))
        
        # Configure grid for input frame
        for i in range(3):
            input_frame.grid_columnconfigure(i, weight=1)
        
        # X matrix input
        ttk.Label(input_frame, text="X Matrix:", font=self.header_font).grid(row=0, column=0, padx=10, pady=5)
        self.x_input = tk.Text(input_frame, height=4, width=20, font=self.input_font)
        self.x_input.grid(row=1, column=0, padx=10, pady=5)
        self.x_input.insert('1.0', "100, 3, 2\n150, 5, 3\n175, 5, 4\n200, 6, 3")
        
        # W matrix input
        ttk.Label(input_frame, text="W Matrix:", font=self.header_font).grid(row=0, column=1, padx=10, pady=5)
        self.w_input = tk.Text(input_frame, height=4, width=20, font=self.input_font)
        self.w_input.grid(row=1, column=1, padx=10, pady=5)
        self.w_input.insert('1.0', "0.1, 0.4\n0.2, 0.5\n0.3, 0.6")
        
        # B vector input
        ttk.Label(input_frame, text="B Vector:", font=self.header_font).grid(row=0, column=2, padx=10, pady=5)
        self.b_input = tk.Text(input_frame, height=2, width=20, font=self.input_font)
        self.b_input.grid(row=1, column=2, padx=10, pady=5)
        self.b_input.insert('1.0', "0.0, 0.0")
        
        # Start button
        start_button = ttk.Button(input_frame, text="Start Computation", command=self.start_computation)
        start_button.grid(row=2, column=0, columnspan=3, pady=20)
        
        # Current step description
        step_frame = ttk.LabelFrame(center_frame, text="Current Step", padding="20")
        step_frame.grid(row=1, column=0, sticky="nsew", pady=20)
        step_frame.grid_columnconfigure(0, weight=1)
        
        self.step_description = tk.Text(step_frame, height=6, width=50, font=self.step_font)
        self.step_description.grid(row=0, column=0, padx=10, pady=10)
        self.step_description.config(state='disabled')
        
        # Result matrix
        result_frame = ttk.LabelFrame(center_frame, text="Result Matrix (Z)", padding="20")
        result_frame.grid(row=2, column=0, sticky="nsew", pady=20)
        result_frame.grid_columnconfigure(0, weight=1)
        
        self.result_text = tk.Text(result_frame, height=8, width=40, font=self.result_font)
        self.result_text.grid(row=0, column=0, padx=10, pady=10)
        self.result_text.config(state='disabled')
        
        # Control frame
        control_frame = ttk.Frame(center_frame)
        control_frame.grid(row=3, column=0, sticky="nsew", pady=20)
        control_frame.grid_columnconfigure(1, weight=1)  # Center column takes extra space
        
        # Control buttons with larger font
        prev_button = ttk.Button(control_frame, text="Previous", command=self.previous_step)
        prev_button.grid(row=0, column=0, padx=10)
        
        self.step_counter = ttk.Label(control_frame, text="Step: 0/0", font=self.header_font)
        self.step_counter.grid(row=0, column=1, padx=10)
        
        next_button = ttk.Button(control_frame, text="Next", command=self.next_step)
        next_button.grid(row=0, column=2, padx=10)

    # [Previous methods remain the same: validate_matrix_input, initialize_computation, 
    # generate_steps, matrix_to_str, update_display, next_step, previous_step]
    def validate_matrix_input(self, input_str, matrix_name):
        try:
            lines = [line.strip() for line in input_str.strip().split('\n') if line.strip()]
            matrix = []
            
            for line in lines:
                values = [val.strip() for val in line.strip('[]').split(',')]
                row = [float(val) for val in values if val]
                if row:
                    matrix.append(row)
            
            if not matrix:
                raise ValueError(f"{matrix_name} cannot be empty")
            
            row_length = len(matrix[0])
            if not all(len(row) == row_length for row in matrix):
                raise ValueError(f"All rows in {matrix_name} must have the same length")
                
            return matrix
            
        except Exception as e:
            messagebox.showerror("Input Error", f"Error in {matrix_name}: {str(e)}")
            return None
            
    def initialize_computation(self):
        try:
            self.X = self.validate_matrix_input(self.x_input.get('1.0', tk.END), "X matrix")
            self.W = self.validate_matrix_input(self.w_input.get('1.0', tk.END), "W matrix")
            
            bias_str = self.b_input.get('1.0', tk.END).strip()
            self.B = [float(x.strip()) for x in bias_str.strip('[]').split(',')]
            
            if not self.X or not self.W or not self.B:
                return False
                
            if len(self.X[0]) != len(self.W):
                messagebox.showerror("Dimension Error", 
                    f"Matrix dimension mismatch: X columns ({len(self.X[0])}) must match W rows ({len(self.W)})")
                return False
                
            if len(self.W[0]) != len(self.B):
                messagebox.showerror("Dimension Error", 
                    f"Matrix dimension mismatch: W columns ({len(self.W[0])}) must match B length ({len(self.B)})")
                return False
            
            self.m = len(self.X)
            self.l = len(self.X[0])
            self.n = len(self.B)
            self.Z = [[0] * self.n for _ in range(self.m)]
            self.steps = []
            self.generate_steps()
            self.current_step = 0
            return True
            
        except Exception as e:
            messagebox.showerror("Input Error", f"Error processing input: {str(e)}")
            return False
    
    def start_computation(self):
        if self.initialize_computation():
            self.update_display()
            
    def generate_steps(self):
        for i in range(self.m):
            for j in range(self.n):
                self.Z[i][j] = self.B[j]
                step = {
                    'description': f'Initialize Z[{i}][{j}] with bias B[{j}] = {self.B[j]}',
                    'i': i,
                    'j': j,
                    'k': None,
                    'calculation': f'Z[{i}][{j}] = {self.B[j]}',
                    'Z': [row[:] for row in self.Z]
                }
                self.steps.append(step)
                
                for k in range(self.l):
                    prev_value = self.Z[i][j]
                    self.Z[i][j] += self.X[i][k] * self.W[k][j]
                    step = {
                        'description': f'Multiply X[{i}][{k}] * W[{k}][{j}] and add to Z[{i}][{j}]',
                        'i': i,
                        'j': j,
                        'k': k,
                        'calculation': f'Z[{i}][{j}] = {prev_value:.4f} + ({self.X[i][k]} * {self.W[k][j]}) = {self.Z[i][j]:.4f}',
                        'Z': [row[:] for row in self.Z]
                    }
                    self.steps.append(step)
    
    def matrix_to_str(self, matrix):
        return '\n'.join([', '.join([f'{x:.4f}' for x in row]) for row in matrix])
        
    def update_display(self):
        if not hasattr(self, 'steps') or not self.steps:
            return
            
        if 0 <= self.current_step < len(self.steps):
            step = self.steps[self.current_step]
            
            self.step_description.config(state='normal')
            self.step_description.delete('1.0', tk.END)
            self.step_description.insert('1.0', f"{step['description']}\n\n{step['calculation']}")
            self.step_description.config(state='disabled')
            
            self.result_text.config(state='normal')
            self.result_text.delete('1.0', tk.END)
            self.result_text.insert('1.0', self.matrix_to_str(step['Z']))
            self.result_text.config(state='disabled')
            
            self.step_counter.config(text=f"Step: {self.current_step + 1}/{len(self.steps)}")
            
    def next_step(self):
        if hasattr(self, 'steps') and self.current_step < len(self.steps) - 1:
            self.current_step += 1
            self.update_display()
            
    def previous_step(self):
        if hasattr(self, 'steps') and self.current_step > 0:
            self.current_step -= 1
            self.update_display()

if __name__ == "__main__":
    root = tk.Tk()
    app = ForwardPassVisualizer(root)
    root.mainloop()