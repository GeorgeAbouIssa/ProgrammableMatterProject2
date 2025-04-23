import sys
import matplotlib.pyplot as plt
import time
import numpy as np
from matplotlib.widgets import Button, RadioButtons, TextBox, Slider
from ConnectedMatterAgent import ConnectedMatterAgent
from Visualizer import Visualizer

cdef class SearchController:
    cdef public tuple grid_size
    cdef public dict formations
    cdef public list start_positions
    cdef public list goal_positions
    cdef public str current_shape
    cdef public str topology
    cdef public double time_limit
    cdef public int max_simultaneous_moves
    cdef public int min_simultaneous_moves
    cdef public bint search_completed
    cdef public double animation_speed
    cdef public object vis
    cdef public object radio_ax
    cdef public object radio
    cdef public object select_button_ax
    cdef public object select_button
    cdef public object grid_text_ax
    cdef public object grid_text_box
    cdef public object grid_button_ax
    cdef public object grid_button
    cdef public object max_moves_panel_ax
    cdef public object sim_moves_slider_ax
    cdef public object sim_moves_slider
    cdef public object min_moves_panel_ax
    cdef public object min_moves_slider_ax
    cdef public object min_moves_slider
    cdef public object anim_speed_panel_ax
    cdef public object anim_speed_slider_ax
    cdef public object anim_speed_slider
    cdef public object agent
    cdef public bint selection_mode
    cdef public list custom_goal
    cdef public bint selection_active
    cdef public bint dragging
    cdef public tuple drag_start
    cdef public list drag_offset
    cdef public int shape_min_x
    cdef public int shape_max_x
    cdef public int shape_min_y
    cdef public int shape_max_y
    cdef public int bound_left
    cdef public int bound_right
    cdef public int bound_top
    cdef public int bound_bottom

    def __init__(self, tuple grid_size, dict formations, str topology="moore", double time_limit=1000, 
                 int max_simultaneous_moves=1, int min_simultaneous_moves=1):
        self.grid_size = grid_size
        self.formations = formations
        self.start_positions = formations["start"]
        self.current_shape = "Ring"
        self.goal_positions = formations["Ring"]
        self.topology = topology
        self.time_limit = time_limit
        self.max_simultaneous_moves = max_simultaneous_moves
        self.min_simultaneous_moves = min(min_simultaneous_moves, max_simultaneous_moves)
        self.search_completed = False

        # Enable interactive mode
        plt.ion()

        # Create visualizer
        self.animation_speed = 0.05
        self.vis = Visualizer(grid_size, [], self.start_positions, self.animation_speed)
        self.vis.draw_grid()
        plt.show(block=False)

        # Add radio buttons
        self.radio_ax = self.vis.fig.add_axes([0.05, 0.05, 0.25, 0.15])
        self.radio = RadioButtons(
            self.radio_ax, 
            labels=list(formations.keys())[1:],
            active=0
        )
        self.radio.on_clicked(self.on_shape_selected)

        # Attach close event
        self.vis.fig.canvas.mpl_connect('close_event', self.on_close)
        
        # Selection mode setup
        self.selection_mode = False
        self.custom_goal = []
        self.selection_active = False
        
        # Add selection button
        self.select_button_ax = self.vis.fig.add_axes([0.4, 0.05, 0.2, 0.075])
        self.select_button = Button(self.select_button_ax, "Select Goal")
        self.select_button.on_clicked(self.toggle_selection_mode)
        
        # Add grid size controls
        label_ax = self.vis.fig.add_axes([0.82, 0.75, 0.15, 0.05])
        label_ax.text(0.5, 0.5, 'Grid Size', ha='center', va='center')
        label_ax.axis('off')
        
        self.grid_text_ax = self.vis.fig.add_axes([0.85, 0.7, 0.09, 0.05])
        self.grid_text_box = TextBox(
            self.grid_text_ax, 
            '',
            initial=f"{grid_size[0]}",
            textalignment='center'
        )
        
        self.grid_button_ax = self.vis.fig.add_axes([0.845, 0.63, 0.1, 0.05])
        self.grid_button = Button(self.grid_button_ax, "Apply")
        self.grid_button.on_clicked(self.change_grid_size)

        # Max simultaneous moves control
        self.max_moves_panel_ax = self.vis.fig.add_axes([0.81, 0.49, 0.17, 0.1])
        self.max_moves_panel_ax.patch.set_facecolor('lightgray')
        self.max_moves_panel_ax.patch.set_alpha(0.3)
        self.max_moves_panel_ax.axis('off')
        
        sim_moves_label_ax = self.vis.fig.add_axes([0.82, 0.55, 0.15, 0.03])
        sim_moves_label_ax.text(0.5, 0.5, 'MAX Simultaneous Moves', 
                               ha='center', va='center', 
                               fontweight='bold', fontsize=9)
        sim_moves_label_ax.axis('off')
        
        self.sim_moves_slider_ax = self.vis.fig.add_axes([0.84, 0.51, 0.1, 0.03])
        self.sim_moves_slider = Slider(
            self.sim_moves_slider_ax, '',
            1, 5, valinit=self.max_simultaneous_moves, valstep=1
        )
        self.sim_moves_slider.on_changed(self.update_max_simultaneous_moves)
        
        # Min simultaneous moves control
        self.min_moves_panel_ax = self.vis.fig.add_axes([0.81, 0.36, 0.17, 0.1])
        self.min_moves_panel_ax.patch.set_facecolor('lightgray')
        self.min_moves_panel_ax.patch.set_alpha(0.3)
        self.min_moves_panel_ax.axis('off')
        
        min_moves_label_ax = self.vis.fig.add_axes([0.82, 0.42, 0.15, 0.03])
        min_moves_label_ax.text(0.5, 0.5, 'MIN Simultaneous Moves', 
                               ha='center', va='center', 
                               fontweight='bold', fontsize=9)
        min_moves_label_ax.axis('off')
        
        self.min_moves_slider_ax = self.vis.fig.add_axes([0.84, 0.38, 0.1, 0.03])
        self.min_moves_slider = Slider(
            self.min_moves_slider_ax, '',
            1, 5, valinit=self.min_simultaneous_moves, valstep=1
        )
        self.min_moves_slider.on_changed(self.update_min_simultaneous_moves)

        # Animation speed control
        self.anim_speed_panel_ax = self.vis.fig.add_axes([0.81, 0.23, 0.17, 0.1])
        self.anim_speed_panel_ax.patch.set_facecolor('lightgray')
        self.anim_speed_panel_ax.patch.set_alpha(0.3)
        self.anim_speed_panel_ax.axis('off')
        
        anim_speed_label_ax = self.vis.fig.add_axes([0.82, 0.29, 0.15, 0.03])
        anim_speed_label_ax.text(0.5, 0.5, 'Animation Speed', 
                                ha='center', va='center', 
                                fontweight='bold', fontsize=9)
        anim_speed_label_ax.axis('off')
        
        self.anim_speed_slider_ax = self.vis.fig.add_axes([0.84, 0.25, 0.1, 0.03])
        self.anim_speed_slider = Slider(
            self.anim_speed_slider_ax, '',
            0.01, 1.0, valinit=self.animation_speed, valfmt='%.2f'
        )
        self.anim_speed_slider.on_changed(self.update_animation_speed)
        
        anim_speed_info_ax = self.vis.fig.add_axes([0.82, 0.21, 0.15, 0.03])
        anim_speed_info_ax.text(0.5, 0.5, 'Seconds per step', 
                               ha='center', va='center', fontsize=8)
        anim_speed_info_ax.axis('off')

        # Print initialization info
        print(f"Initializing Connected Programmable Matter Agent...")
        print(f"Grid size: {grid_size}")
        print(f"Start positions: {self.start_positions}")
        print(f"Current shape: {self.current_shape}")
        print(f"Topology: {topology}")
        print(f"Time limit: {time_limit} seconds")
        print(f"Constraint: All elements must remain connected during movement")

        # Initialize the agent
        self.agent = ConnectedMatterAgent(
            grid_size, 
            self.start_positions, 
            self.goal_positions, 
            topology,
            max_simultaneous_moves=self.max_simultaneous_moves,
            min_simultaneous_moves=self.min_simultaneous_moves
        )
        
        # Set up search button
        self.vis.button.on_clicked(self.handle_button)
        
        # Dragging setup
        self.dragging = False
        self.drag_start = None
        self.drag_offset = None
        
        # Boundary checking
        self.shape_min_x = 0
        self.shape_max_x = 0
        self.shape_min_y = 0
        self.shape_max_y = 0
        
        # Add mouse events
        self.vis.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.vis.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.vis.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def toggle_selection_mode(self, event):
        """Toggle between normal mode and goal selection mode"""
        if self.dragging:  # Don't allow mode switch while dragging
            return
            
        self.selection_mode = not self.selection_mode
        self.selection_active = self.selection_mode
        
        if self.selection_mode:
            # Entering selection mode
            self.custom_goal = []  # Clear any previous selection
            self.select_button.label.set_text("Confirm Goal")
            self.vis.update_text("Click on grid cells to define goal state", color="blue")
            
            # Clear any highlighted goal shape
            self.vis.draw_grid(highlight_goal=False)
        else:
            # Exiting selection mode
            # MODIFIED: Remove constraint that custom goal must have same number of blocks as start
            if len(self.custom_goal) > 0:  # Just require at least one block
                self.goal_positions = self.custom_goal.copy()  # Make a copy of the custom goal
                self.agent = ConnectedMatterAgent(
                    self.grid_size, 
                    self.start_positions, 
                    self.goal_positions, 
                    self.topology,
                    max_simultaneous_moves=self.max_simultaneous_moves,
                    min_simultaneous_moves=self.min_simultaneous_moves
                )
                self.select_button.label.set_text("Select Goal")
                
                # Update visualization with the new goal shape
                self.vis.draw_grid()
                self.vis.highlight_goal_shape(self.goal_positions)
                
                if len(self.goal_positions) < len(self.start_positions):
                    self.vis.update_text(f"Custom goal set with {len(self.goal_positions)} blocks (fewer than start)", color="green")
                else:
                    self.vis.update_text(f"Custom goal set with {len(self.goal_positions)} blocks", color="green")
                
                # Reset search state
                self.search_completed = False
                self.vis.animation_started = False
                self.vis.animation_done = False
                self.vis.current_step = 0
                self.vis.path = None
            else:
                self.selection_mode = True  # Stay in selection mode if invalid
                self.select_button.label.set_text("Select Goal")
                self.vis.update_text(f"Invalid goal: Need at least 1 block", color="red")

    def on_grid_click(self, event, int x, int y):
        """Handle grid cell clicks for goal selection"""
        if not self.selection_mode or not self.selection_active:
            return
        
        pos = (x, y)
        
        if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
            if pos in self.custom_goal:
                # Remove this position
                self.custom_goal.remove(pos)
            else:
                # Add this position
                # MODIFIED: Remove the limit on maximum number of blocks
                self.custom_goal.append(pos)
            
            # Redraw the grid with current selection
            self.vis.draw_grid(highlight_goal=False)
            for cell_pos in self.custom_goal:
                self.highlight_cell(cell_pos, color='green')
            
            # Update the counter
            self.vis.update_text(f"Selected {len(self.custom_goal)} block(s)", color="blue")
    
    def highlight_cell(self, tuple pos, str color='green'):
        """Highlight a grid cell with the specified color"""
        cdef int x, y
        x, y = pos
        self.vis.ax.add_patch(plt.Rectangle((y, x), 1, 1, color=color, alpha=0.7))
        plt.draw()    

    def on_close(self, event):
        """Stops execution when the Matplotlib window is closed."""
        print("\nWindow closed. Exiting program.")
        sys.exit()  # Forcefully stop the script
        
    def on_shape_selected(self, str label):
        """Handle shape selection from radio buttons"""
        # Cancel selection mode if active when changing shapes
        if self.selection_mode:
            self.selection_mode = False
            self.selection_active = False
            self.select_button.label.set_text("Select Goal")
        
        self.current_shape = label
        self.goal_positions = self.formations[self.current_shape]
        
        # Update agent with new goal positions
        self.agent = ConnectedMatterAgent(
            self.grid_size, 
            self.start_positions, 
            self.goal_positions, 
            self.topology,
            max_simultaneous_moves=self.max_simultaneous_moves,
            min_simultaneous_moves=self.min_simultaneous_moves
        )
        
        print(f"Selected shape: {self.current_shape}")
        print(f"Goal positions: {self.goal_positions}")
        
        # Reset search state
        self.search_completed = False
        self.vis.animation_started = False
        self.vis.animation_done = False
        self.vis.current_step = 0
        self.vis.path = None
        
        # Update visualization with highlighted goal shape
        self.vis.draw_grid()
        self.vis.highlight_goal_shape(self.goal_positions)
        self.vis.button.label.set_text("Search")
        
        # Display message about number of blocks
        if len(self.goal_positions) < len(self.start_positions):
            self.vis.update_text(f"Selected {self.current_shape} shape ({len(self.goal_positions)} blocks, fewer than start)", color="blue")
        else:
            self.vis.update_text(f"Selected {self.current_shape} shape ({len(self.goal_positions)} blocks)", color="blue")

    def handle_button(self, event):
        """Handle button clicks based on current state"""
        if not self.search_completed:
            self.run_search(event)
        else:
            self.vis.handle_button_click(event)

    def run_search(self, event):
        """Runs the search when the Search button is clicked."""
        # Clear goal shape highlight
        self.vis.draw_grid()
        self.vis.update_text("Searching for a path...", color="red")
        plt.pause(1)  # Force update to show "Searching..." before search starts
        
        print("\nSearching for optimal path with connectivity constraint...")
        cdef double start_time = time.time()
        path = self.agent.search(self.time_limit)
        cdef double search_time = time.time() - start_time
        
        self.search_completed = True

        if path:
            print(f"Path found with {len(path)-1} moves in {search_time:.2f} seconds")
            print(f"Ready for visualization...")
            self.vis.animation_speed = self.animation_speed  # Ensure animation speed is updated
            self.vis.path = path  # Update path in visualizer
            self.vis.button.label.set_text("Start")  # Set button text to Start
            self.vis.update_text(f"Path found ({len(path)-1} moves)", color="green")
            plt.draw()
        else:
            print(f"No path found after {search_time:.2f} seconds")
            self.vis.button.label.set_text("Search")  # Reset button text
            self.vis.update_text("No paths found", color="red")
            plt.draw()

    def update_max_simultaneous_moves(self, val):
        """Update the maximum number of simultaneous moves"""
        self.max_simultaneous_moves = int(val)
        
        # Ensure min doesn't exceed max
        if self.min_simultaneous_moves > self.max_simultaneous_moves:
            self.min_simultaneous_moves = self.max_simultaneous_moves
            self.min_moves_slider.set_val(self.min_simultaneous_moves)
            
        self.vis.update_text(f"Simultaneous moves: {self.min_simultaneous_moves}-{self.max_simultaneous_moves}", color="blue")
        
        # Update agent with new parameter
        self.agent = ConnectedMatterAgent(
            self.grid_size, 
            self.start_positions, 
            self.goal_positions, 
            self.topology,
            max_simultaneous_moves=self.max_simultaneous_moves,
            min_simultaneous_moves=self.min_simultaneous_moves
        )
        
        # Reset search state
        self.search_completed = False
        self.vis.animation_started = False
        self.vis.animation_done = False
        self.vis.current_step = 0
        self.vis.path = None
    
    def update_min_simultaneous_moves(self, val):
        """Update the minimum number of simultaneous moves"""
        self.min_simultaneous_moves = int(val)
        
        # Ensure min doesn't exceed max
        if self.min_simultaneous_moves > self.max_simultaneous_moves:
            self.max_simultaneous_moves = self.min_simultaneous_moves
            self.sim_moves_slider.set_val(self.max_simultaneous_moves)
            
        self.vis.update_text(f"Simultaneous moves: {self.min_simultaneous_moves}-{self.max_simultaneous_moves}", color="blue")
        
        # Update agent with new parameter
        self.agent = ConnectedMatterAgent(
            self.grid_size, 
            self.start_positions, 
            self.goal_positions, 
            self.topology,
            max_simultaneous_moves=self.max_simultaneous_moves,
            min_simultaneous_moves=self.min_simultaneous_moves
        )
        
        # Reset search state
        self.search_completed = False
        self.vis.animation_started = False
        self.vis.animation_done = False
        self.vis.current_step = 0
        self.vis.path = None

    def update_animation_speed(self, val):
        """Update the animation speed (seconds between frames)"""
        self.animation_speed = val
        self.vis.set_animation_speed(val)
        self.vis.update_text(f"Animation speed: {val:.2f} sec/step", color="blue")

    def on_text_submit(self, text):
        """Handle grid size text submission"""
        self.change_grid_size(None)  # Call change_grid_size when Enter is pressed
            
    def change_grid_size(self, event):
        """Handle grid size change"""
        try:
            n = int(self.grid_text_box.text)
            if n < 10:
                self.vis.update_text("Grid size must be at least 10", color="red")
                return
            if n > 200:
                self.vis.update_text("Grid size cannot exceed 200", color="red")
                return

            # Update grid size
            self.grid_size = (n, n)
            
            # Update text box to show clean number
            self.grid_text_box.set_val(str(n))
            
            # Reinitialize agent with new grid size
            self.agent = ConnectedMatterAgent(
                self.grid_size, 
                self.start_positions, 
                self.goal_positions, 
                self.topology,
                max_simultaneous_moves=self.max_simultaneous_moves,
                min_simultaneous_moves=self.min_simultaneous_moves
            )
            
            # Update visualization
            self.vis.grid_size = self.grid_size
            self.vis.draw_grid()
            self.vis.update_text(f"Grid size updated to {n}x{n}", color="green")
            
            # Reset search state
            self.search_completed = False
            self.vis.animation_started = False
            self.vis.animation_done = False
            self.vis.current_step = 0
            self.vis.path = None
            
        except ValueError:
            self.vis.update_text("Invalid grid size. Enter a number between 10-200", color="red")

    def on_mouse_press(self, event):
        """Handle mouse press events for dragging and selection"""
        cdef int x, y
        
        if event.inaxes != self.vis.ax:
            return
            
        # Convert click coordinates to grid cell
        x = int(event.ydata)
        y = int(event.xdata)
        
        # Handle selection mode separately from dragging
        if self.selection_mode and self.selection_active:
            self.on_grid_click(event, x, y)
            return
        
        # Only allow dragging when not in selection mode
        click_pos = (x, y)
        
        # Check if click is within the goal shape
        if any(abs(gx - x) < 1 and abs(gy - y) < 1 for gx, gy in self.goal_positions):
            self.dragging = True
            self.drag_start = click_pos
            self.drag_offset = []
            
            # Calculate offsets for all points relative to click position
            for gx, gy in self.goal_positions:
                self.drag_offset.append((gx - x, gy - y))
            
            # Calculate shape boundaries for edge checking
            x_coords = [pos[0] for pos in self.goal_positions]
            y_coords = [pos[1] for pos in self.goal_positions]
            self.shape_min_x = min(x_coords)
            self.shape_max_x = max(x_coords)
            self.shape_min_y = min(y_coords)
            self.shape_max_y = max(y_coords)
            
            # Calculate shape dimensions
            shape_width = self.shape_max_x - self.shape_min_x
            shape_height = self.shape_max_y - self.shape_min_y
            
            # Calculate distance from click to shape bounds
            self.bound_left = x - self.shape_min_x
            self.bound_right = self.shape_max_x - x
            self.bound_top = y - self.shape_min_y
            self.bound_bottom = self.shape_max_y - y
    
    def on_mouse_release(self, event):
        """Handle mouse release events for dragging"""
        cdef int x, y, new_x, new_y
        
        if self.dragging and not self.selection_mode:
            self.dragging = False
            if event.inaxes == self.vis.ax:
                # Snap to grid
                x = int(event.ydata)
                y = int(event.xdata)
                # Apply boundary constraints
                x, y = self.constrain_to_boundaries(x, y)
                
                new_positions = []
                for offset_x, offset_y in self.drag_offset:
                    new_x = x + offset_x
                    new_y = y + offset_y
                    if 0 <= new_x < self.grid_size[0] and 0 <= new_y < self.grid_size[1]:
                        new_positions.append((new_x, new_y))
                
                if len(new_positions) == len(self.goal_positions):
                    self.goal_positions = new_positions
                    self.agent = ConnectedMatterAgent(
                        self.grid_size, 
                        self.start_positions, 
                        self.goal_positions, 
                        self.topology,
                        max_simultaneous_moves=self.max_simultaneous_moves,
                        min_simultaneous_moves=self.min_simultaneous_moves
                    )
                    self.search_completed = False
                    self.vis.draw_grid()
                    self.vis.highlight_goal_shape(self.goal_positions)
                    self.vis.update_text("Goal shape moved", color="blue")
    
    def constrain_to_boundaries(self, int x, int y):
        """Constrain the drag point to keep the shape within grid boundaries"""
        # Constrain x-coordinate
        cdef int min_x = self.bound_left  # Minimum allowed x (to keep left edge in bounds)
        cdef int max_x = self.grid_size[0] - 1 - self.bound_right  # Maximum allowed x (to keep right edge in bounds)
        x = max(min_x, min(x, max_x))
        
        # Constrain y-coordinate
        cdef int min_y = self.bound_top  # Minimum allowed y (to keep top edge in bounds)
        cdef int max_y = self.grid_size[1] - 1 - self.bound_bottom  # Maximum allowed y (to keep bottom edge in bounds)
        y = max(min_y, min(y, max_y))
        
        return x, y
    
    def on_mouse_move(self, event):
        """Handle mouse movement events for dragging"""
        cdef double x, y
        cdef int xi, yi, new_x, new_y
        
        if self.dragging and not self.selection_mode and event.inaxes == self.vis.ax:
            # Get cursor position with boundary constraints
            x = event.ydata
            y = event.xdata
            xi = int(x)
            yi = int(y)
            xi, yi = self.constrain_to_boundaries(xi, yi)
            
            # Clear and redraw the grid
            self.vis.draw_grid()
            
            # Draw the shape at the new position
            temp_positions = []
            for offset_x, offset_y in self.drag_offset:
                new_x = xi + offset_x
                new_y = yi + offset_y
                temp_positions.append((new_x, new_y))
            
            # Highlight the shape at its temporary position
            self.vis.highlight_goal_shape(temp_positions)