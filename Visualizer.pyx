import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

cdef class Visualizer:
    cdef public object fig
    cdef public object ax
    cdef public object button_ax
    cdef public object button
    cdef public object text_annotation
    cdef public tuple grid_size
    cdef public list path
    cdef public list start_positions
    cdef public double animation_speed
    cdef public bint paused
    cdef public bint animation_done
    cdef public int current_step
    cdef public bint animation_started

    def __init__(self, tuple grid_size, list path, list start_positions, double animation_speed=0.05):
        self.grid_size = grid_size
        self.path = path
        self.start_positions = start_positions
        self.animation_speed = animation_speed
        self.paused = False
        self.animation_done = False
        self.current_step = 0
        self.animation_started = False

        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.fig.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.95)

        # Create button
        self.button_ax = self.fig.add_axes([0.7, 0.05, 0.2, 0.075])
        self.button = Button(self.button_ax, "Search")
        
        # Initialize text annotation for status messages
        self.text_annotation = self.ax.text(
            self.grid_size[1] / 2, self.grid_size[0] + 0.3, "", 
            ha="center", fontsize=12, fontweight="bold"
        )

        self.draw_grid()

    def draw_grid(self, bint highlight_initial=True, bint highlight_goal=False):
        """Draws the grid and optionally highlights positions."""
        self.ax.clear()
        self.ax.set_xticks(np.arange(self.grid_size[1] + 1), minor=False)
        self.ax.set_yticks(np.arange(self.grid_size[0] + 1), minor=False)
        self.ax.grid(which="major", color="black", linestyle='-', linewidth=0.5)
        self.ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        self.ax.set_aspect('equal')
        self.ax.set_xlim(0, self.grid_size[1])
        self.ax.set_ylim(0, self.grid_size[0])

        if highlight_initial:
            for x, y in self.start_positions:
                self.ax.add_patch(plt.Rectangle((y, x), 1, 1, color='Grey', label="Start"))
    
        self.text_annotation = self.ax.text(
            self.grid_size[1] / 2, self.grid_size[0] + 0.3, 
            self.text_annotation.get_text(), ha="center", fontsize=12, fontweight="bold"
        )
        plt.draw()
        
    def update_text(self, str message, str color="black"):
        """Updates the status message above the grid dynamically with color."""
        self.text_annotation.set_text(message)
        self.text_annotation.set_color(color)
        plt.draw()

    def handle_button_click(self, event):
        """Handles the button click event for start, pause, resume, and restart."""
        if not self.path:
            print("No path found. Make sure path is set before starting animation.")
            return

        if self.animation_done:
            # Restart the animation
            self.animation_done = False
            self.animation_started = False
            self.current_step = 0
            self.button.label.set_text("Start")
            self.update_text("Select a shape for search.", color="blue")
        elif not self.animation_started:
            # Start animation for the first time
            self.animation_started = True
            self.button.label.set_text("Pause")
            self.update_text("Animating...", color="black")
            self.animate_path()
        else:
            # Toggle pause state
            self.paused = not self.paused
            if self.paused:
                self.button.label.set_text("Resume")
                self.update_text("Paused", color="orange")
            else:
                self.button.label.set_text("Pause")
                self.update_text("Animating...", color="black")
                self.animate_path()

    def animate_path(self):
        """Animates the path step by step."""
        if not self.path:
            self.update_text("No paths found", color="red")
            return

        # Continue from the current step instead of restarting
        while self.current_step < len(self.path):
            if self.paused:
                return  # Stop animation if paused

            # Draw the current step
            self.ax.clear()  # Clear the grid
            self.ax.set_xticks(np.arange(self.grid_size[1] + 1), minor=False)
            self.ax.set_yticks(np.arange(self.grid_size[0] + 1), minor=False)
            self.ax.grid(which="major", color="black", linestyle='-', linewidth=0.5)
            self.ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
            self.ax.set_aspect('equal')
            self.ax.set_xlim(0, self.grid_size[1])
            self.ax.set_ylim(0, self.grid_size[0])
        
            # Restore text annotation
            self.text_annotation = self.ax.text(
                self.grid_size[1] / 2, self.grid_size[0] + 0.3, 
                "Animating...", ha="center", fontsize=12, fontweight="bold"
            )

            # Get current positions in the path
            positions = self.path[self.current_step]
        
            # Check for overlaps
            if len(set(positions)) < len(positions):
                self.update_text("Warning: Node overlap detected!", color="red")
        
            # Draw each position without labels
            for pos in positions:
                x, y = pos
                self.ax.add_patch(plt.Rectangle((y, x), 1, 1, color='grey'))
        
            plt.pause(self.animation_speed)
            self.current_step += 1

        # Animation completed
        if self.current_step >= len(self.path):
            self.animation_done = True
            self.button.label.set_text("Restart")
            self.update_text("Start another search.", color="green")

        plt.draw()

    def highlight_goal_shape(self, list goal_positions):
        """Highlight the goal shape with a dim green color"""
        cdef float x, y
        for pos in goal_positions:
            x, y = pos
            rect = plt.Rectangle((y, x), 1, 1, color='green', alpha=0.3)
            self.ax.add_patch(rect)
        plt.draw()

    def set_animation_speed(self, double speed):
        """Update the animation speed"""
        self.animation_speed = speed