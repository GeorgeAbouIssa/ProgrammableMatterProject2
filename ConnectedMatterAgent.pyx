import heapq
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import multiprocessing as mp
from functools import partial
import copy

cdef class ConnectedMatterAgent:
    # Keep all existing class attributes
    cdef public tuple grid_size
    cdef public list start_positions
    cdef public list goal_positions
    cdef public str topology
    cdef public int max_simultaneous_moves
    cdef public int min_simultaneous_moves
    cdef public list directions
    cdef public object start_state
    cdef public object goal_state
    cdef public tuple goal_centroid
    cdef public dict valid_moves_cache
    cdef public dict articulation_points_cache
    cdef public dict connectivity_check_cache
    cdef public int beam_width
    cdef public int max_iterations
    cdef public object target_state  # Subset of blocks to be used for goal
    cdef public object non_target_state  # Blocks that won't move to goal
    cdef public bint allow_disconnection  # Flag to allow blocks to disconnect
    cdef public list target_block_list  # List of blocks that will be moved
    cdef public list fixed_block_list  # List of blocks that won't be moved
    cdef public list goal_components  # List of connected components in goal
    cdef public list goal_component_centroids  # Centroids for each goal component
    cdef public dict block_component_assignment  # Maps target blocks to goal components
    cdef public bint multi_component_goal  # Flag indicating goal has multiple components
    cdef public bint use_multiprocessing  # New flag to enable multiprocessing
    cdef public int num_processes  # Number of processes to use for parallel search

# Full __init__ method with proper indentation
    def __init__(self, tuple grid_size, list start_positions, list goal_positions, str topology="moore", 
                 int max_simultaneous_moves=1, int min_simultaneous_moves=1, bint use_multiprocessing=True):
        self.grid_size = grid_size
        self.start_positions = list(start_positions)
        self.goal_positions = list(goal_positions)
        self.topology = topology
        self.max_simultaneous_moves = max_simultaneous_moves
        self.min_simultaneous_moves = min(min_simultaneous_moves, max_simultaneous_moves)  # Ensure min <= max
        
        # Add multiprocessing properties
        self.use_multiprocessing = use_multiprocessing
        self.num_processes = min(mp.cpu_count(), 4)  # Limit to a reasonable number
        
        # Set moves based on topology
        if self.topology == "moore":
            self.directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        else:  # Von Neumann
            self.directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        
        # Ensure no duplicates in goal positions
        unique_goal_positions = []
        goal_positions_set = set()
        for pos in goal_positions:
            if pos not in goal_positions_set:
                goal_positions_set.add(pos)
                unique_goal_positions.append(pos)
                
        if len(unique_goal_positions) != len(goal_positions):
            print("WARNING: Duplicate positions detected in goal state. Removed duplicates.")
            self.goal_positions = unique_goal_positions
            
        # Initialize the start and goal states
        self.start_state = frozenset((x, y) for x, y in self.start_positions)
        self.goal_state = frozenset((x, y) for x, y in self.goal_positions)
        
        # Analyze the goal state for connected components
        self.goal_components = self.find_connected_components(self.goal_positions)
        self.multi_component_goal = len(self.goal_components) > 1
        
        # CRITICAL FIX: Force disconnection to be allowed for multi-component goals
        if self.multi_component_goal:
            self.allow_disconnection = True
            print(f"Multiple disconnected components detected - enabling component separation")
            # Calculate centroid for each component
            self.goal_component_centroids = [self.calculate_centroid(component) 
                                            for component in self.goal_components]
            for i, (component, centroid) in enumerate(zip(self.goal_components, self.goal_component_centroids)):
                print(f"Component {i+1}: {len(component)} blocks, centroid at {centroid}")
        else:
            # Default for single-component goals
            self.allow_disconnection = False
            self.goal_component_centroids = []
        
        # Calculate the centroid of all goal positions (still useful for overall movement)
        self.goal_centroid = self.calculate_centroid(self.goal_positions)
        
        # Handle cases where goal has fewer blocks than start
        if len(self.goal_positions) < len(start_positions):
            # Flag to allow disconnection when goal has fewer blocks
            self.allow_disconnection = True
            
            if self.multi_component_goal:
                # Select blocks for each component separately based on proximity to component centroids
                self.target_block_list = self.select_blocks_for_components()
                
                # CRITICAL FIX: Verify block assignment
                if len(self.target_block_list) != len(self.goal_positions):
                    print(f"WARNING: Selected {len(self.target_block_list)} blocks but need {len(self.goal_positions)}")
                    # Try to fix by selecting exactly the right number of blocks
                    if len(self.target_block_list) < len(self.goal_positions):
                        # Find more blocks to add
                        additional_needed = len(self.goal_positions) - len(self.target_block_list)
                        print(f"Attempting to find {additional_needed} additional blocks")
                        
                        # Find unassigned blocks
                        unassigned = [pos for pos in self.start_positions if pos not in self.target_block_list]
                        # Add blocks up to the needed amount
                        self.target_block_list.extend(unassigned[:additional_needed])
                        
                        # Assign these to components that need more blocks
                        for i, pos in enumerate(unassigned[:additional_needed]):
                            # Find component with fewest blocks
                            component_counts = {}
                            for p, comp_idx in self.block_component_assignment.items():
                                component_counts[comp_idx] = component_counts.get(comp_idx, 0) + 1
                            
                            target_comp = min(range(len(self.goal_components)), 
                                             key=lambda c: component_counts.get(c, 0))
                            self.block_component_assignment[pos] = target_comp
                            print(f"Assigned additional block at {pos} to Component {target_comp+1}")
                    elif len(self.target_block_list) > len(self.goal_positions):
                        # Remove excess blocks
                        excess = len(self.target_block_list) - len(self.goal_positions)
                        print(f"Removing {excess} excess blocks")
                        removed = self.target_block_list[len(self.goal_positions):]
                        self.target_block_list = self.target_block_list[:len(self.goal_positions)]
                        
                        # Remove these from block_component_assignment
                        for pos in removed:
                            if pos in self.block_component_assignment:
                                del self.block_component_assignment[pos]
            else:
                # Standard selection for single-component goals
                self.target_block_list = self.select_closest_blocks_to_goal()
                
            self.fixed_block_list = [pos for pos in self.start_positions if pos not in self.target_block_list]
            
            # Convert to frozensets for state operations
            self.target_state = frozenset((x, y) for x, y in self.target_block_list)
            self.non_target_state = frozenset((x, y) for x, y in self.fixed_block_list)
            
            print(f"Goal has fewer blocks ({len(self.goal_positions)}) than start ({len(start_positions)})")
            print(f"Selected {len(self.target_block_list)} blocks closest to the goal")
            print(f"Blocks will be allowed to disconnect during movement")
            print(f"Fixed blocks: {len(self.fixed_block_list)} will remain stationary")
        else:
            # If goal has same or more blocks, all start blocks are target blocks
            self.target_block_list = self.start_positions.copy()
            self.fixed_block_list = []
            self.target_state = self.start_state
            self.non_target_state = frozenset()
            # No component assignment needed when all blocks are targets
            self.block_component_assignment = {}
        
        # Cache for valid moves to avoid recomputation
        self.valid_moves_cache = {}
        
        # For optimizing the search
        self.articulation_points_cache = {}
        self.connectivity_check_cache = {}
        
        # Enhanced parameters for improved search
        self.beam_width = 500  # Increased beam width for better exploration
        self.max_iterations = 10000  # Limit iterations to prevent infinite loops
        
        # For multi-component goals, do additional analysis
        if self.multi_component_goal and self.allow_disconnection:
            self._analyze_goal_components()
            
        # CRITICAL FIX: Validate component assignments
        if self.multi_component_goal and self.allow_disconnection:
            self._validate_component_assignments()

    def _analyze_goal_components(self):
        """Analyze goal components for better planning"""
        print("Analyzing disconnected goal components...")
        
        # Validate component assignments
        for i, component in enumerate(self.goal_components):
            assigned_blocks = [pos for pos in self.target_block_list 
                              if self.block_component_assignment.get(pos) == i]
            if len(assigned_blocks) != len(component):
                print(f"WARNING: Component {i+1} has {len(assigned_blocks)} blocks assigned but needs {len(component)}")
        
        # Check distances between components
        for i, centroid1 in enumerate(self.goal_component_centroids):
            for j, centroid2 in enumerate(self.goal_component_centroids):
                if i < j:
                    dist = abs(centroid1[0] - centroid2[0]) + abs(centroid1[1] - centroid2[1])
                    print(f"Distance between Component {i+1} and {j+1}: {dist}")
                    
                    # Warn about potential collisions
                    if dist < 3:
                        print(f"WARNING: Components {i+1} and {j+1} are very close - may be difficult to separate")

    def _validate_component_assignments(self):
        """
        Validate that component assignments are correct
        - Each component should have the right number of blocks
        - Blocks should be assigned to appropriate components
        """
        if not self.multi_component_goal:
            return
            
        print("Validating component assignments...")
        
        # Check if each component has the right number of blocks
        component_counts = {}
        for pos, comp_idx in self.block_component_assignment.items():
            component_counts[comp_idx] = component_counts.get(comp_idx, 0) + 1
        
        for i, component in enumerate(self.goal_components):
            expected = len(component)
            actual = component_counts.get(i, 0)
            
            if expected != actual:
                print(f"  WARNING: Component {i+1} has {actual} blocks assigned but needs {expected}")
                # Try to fix by reassigning blocks if needed
                if actual < expected:
                    print(f"  Attempting to find additional blocks for Component {i+1}")
                    self._find_additional_blocks_for_component(i, expected - actual)
        
        print("Component assignment validation complete")

    def _find_additional_blocks_for_component(self, comp_idx, blocks_needed):
        """
        Find additional blocks for a component that doesn't have enough
        """
        # Get already assigned blocks
        assigned_blocks = set(self.target_block_list)
        component_centroid = self.goal_component_centroids[comp_idx]
        
        # Find closest unassigned blocks
        unassigned_blocks = [pos for pos in self.start_positions if pos not in assigned_blocks]
        block_distances = []
        
        for pos in unassigned_blocks:
            # Calculate distance to this component's centroid
            dist = abs(pos[0] - component_centroid[0]) + abs(pos[1] - component_centroid[1])
            block_distances.append((dist, pos))
        
        # Sort by distance
        block_distances.sort()
        
        # Add blocks to the component
        added_blocks = 0
        for _, pos in block_distances:
            # Add this block to the target list
            self.target_block_list.append(pos)
            # Update the component assignment
            self.block_component_assignment[pos] = comp_idx
            # Remove from fixed blocks if it's there
            if pos in self.fixed_block_list:
                self.fixed_block_list.remove(pos)
            
            added_blocks += 1
            print(f"  Added block at {pos} to Component {comp_idx+1}")
            
            if added_blocks >= blocks_needed:
                break
        
        # Update the target and non-target states
        self.target_state = frozenset((x, y) for x, y in self.target_block_list)
        self.non_target_state = frozenset((x, y) for x, y in self.fixed_block_list)

    def find_connected_components(self, positions):
        """Find all connected components in a set of positions"""
        if not positions:
            return []
            
        positions_set = set(positions)
        components = []
        
        while positions_set:
            # Take an unvisited position as the start of a new component
            start = next(iter(positions_set))
            
            # Find all positions connected to start using BFS
            component = set()
            visited = {start}
            queue = deque([start])
            
            while queue:
                current = queue.popleft()
                component.add(current)
                
                # Check all adjacent positions
                for dx, dy in self.directions:
                    neighbor = (current[0] + dx, current[1] + dy)
                    if neighbor in positions_set and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            # Add the component to the list
            components.append(list(component))
            
            # Remove the positions in this component from the unvisited set
            positions_set -= component
        
        # Sort components by size (largest first)
        components.sort(key=len, reverse=True)
        return components
    
    def select_blocks_for_components(self):
        """
        Enhanced method to select blocks for each component in multi-component goals
        """
        print("Selecting blocks for multiple goal components...")
        
        # Initialize component assignments
        component_blocks = {i: [] for i in range(len(self.goal_components))}
        block_component_assignment = {}
        
        # Calculate centroids for each component
        component_centroids = [self.calculate_centroid(component) for component in self.goal_components]
        
        # First pass: Assign minimum blocks to each component
        remaining_blocks = list(self.start_positions)
        
        for comp_idx, component in enumerate(self.goal_components):
            required_blocks = len(component)
            centroid = component_centroids[comp_idx]
            
            # Sort blocks by distance to this component's centroid
            block_distances = []
            for pos in remaining_blocks:
                dist = abs(pos[0] - centroid[0]) + abs(pos[1] - centroid[1])
                block_distances.append((dist, pos))
            
            block_distances.sort()  # Sort by distance
            
            # Assign blocks to this component
            for i in range(min(required_blocks, len(block_distances))):
                _, pos = block_distances[i]
                component_blocks[comp_idx].append(pos)
                block_component_assignment[pos] = comp_idx
                remaining_blocks.remove(pos)
                
            print(f"  Assigned {len(component_blocks[comp_idx])}/{required_blocks} blocks to Component {comp_idx+1}")
        
        # Add any unassigned blocks to components that need more
        for comp_idx, blocks in component_blocks.items():
            required = len(self.goal_components[comp_idx])
            if len(blocks) < required and remaining_blocks:
                # How many more blocks needed
                needed = required - len(blocks)
                # Take up to that many from remaining blocks
                additional = remaining_blocks[:needed]
                
                # Add to this component
                for pos in additional:
                    component_blocks[comp_idx].append(pos)
                    block_component_assignment[pos] = comp_idx
                    remaining_blocks.remove(pos)
                    
                print(f"  Added {len(additional)} more blocks to Component {comp_idx+1}")
        
        # Store the component assignment for later use
        self.block_component_assignment = block_component_assignment
        
        # Return all blocks selected for the goal
        selected_blocks = []
        for blocks in component_blocks.values():
            selected_blocks.extend(blocks)
        
        return selected_blocks

# New method for parallel component search
    def search_parallel_components(self, time_limit=30):
        """Run parallel searches for each component"""
        print(f"Starting parallel search for {len(self.goal_components)} disconnected components...")
        
        # Make sure component assignments are valid
        self._validate_component_assignments()
        
        # Create component tasks
        component_tasks = []
        for comp_idx in range(len(self.goal_components)):
            # Get blocks assigned to this component
            component_blocks = [pos for pos in self.target_block_list 
                            if self.block_component_assignment.get(pos) == comp_idx]
            
            if not component_blocks:
                print(f"No blocks assigned to Component {comp_idx+1}, skipping")
                continue
                
            # Add task
            component_tasks.append((comp_idx, component_blocks))
        
        # Allocate time for each component
        component_time_limit = time_limit / max(1, len(component_tasks))
        
        # Run searches sequentially
        component_results = []
        for comp_idx, blocks in component_tasks:
            print(f"Searching for Component {comp_idx+1} with {len(blocks)} blocks")
            result = self.search_single_component(comp_idx, self.start_state, blocks, component_time_limit)
            component_results.append(result)
            print(f"Completed search for Component {comp_idx+1}, path length: {len(result[1]) if result and result[1] else 0}")
        
        # Combine results
        merged_path = self.merge_component_paths(component_results)
        
        # Make sure all blocks are in the final path
        if merged_path:
            final_state = merged_path[-1]
            expected_count = len(self.target_block_list)
            actual_count = len(final_state)
            
            if actual_count < expected_count:
                print(f"WARNING: Final state has {actual_count} blocks but expected {expected_count}")
        
        return merged_path

    def select_closest_blocks_to_goal(self):
        """
        Select blocks from start state that are closest to the goal centroid
        Returns a list of selected blocks
        """
        # Calculate distances from each start position to goal centroid
        distances = []
        for pos in self.start_positions:
            # Manhattan distance to centroid
            dist = abs(pos[0] - self.goal_centroid[0]) + abs(pos[1] - self.goal_centroid[1])
            distances.append((dist, pos))
        
        # Sort by distance (ascending)
        distances.sort()
        
        # Select the number of blocks needed for the goal
        selected_blocks = [pos for _, pos in distances[:len(self.goal_positions)]]
        
        # For single component, simple 1:1 mapping
        self.block_component_assignment = {pos: 0 for pos in selected_blocks}
        
        return selected_blocks
        
    def calculate_centroid(self, positions):
        """Calculate the centroid (average position) of a set of positions"""
        cdef double x_sum, y_sum
        
        if not positions:
            return (0, 0)
        x_sum = sum(pos[0] for pos in positions)
        y_sum = sum(pos[1] for pos in positions)
        return (x_sum / len(positions), y_sum / len(positions))
    
    def is_connected(self, positions):
        """Check if all positions are connected using BFS"""
        if not positions:
            return True
            
        # Use cache if available
        positions_hash = hash(frozenset(positions))
        if positions_hash in self.connectivity_check_cache:
            return self.connectivity_check_cache[positions_hash]
            
        # Convert to set for O(1) lookup
        positions_set = set(positions)
        
        # Start BFS from first position
        start = next(iter(positions_set))
        visited = {start}
        queue = deque([start])
        
        while queue:
            current = queue.popleft()
            
            # Check all adjacent positions
            for dx, dy in self.directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor in positions_set and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # All positions should be visited if connected
        is_connected = len(visited) == len(positions_set)
        
        # Cache the result
        self.connectivity_check_cache[positions_hash] = is_connected
        return is_connected
    
# Improved check_component_connectivity method
    def check_component_connectivity(self, state):
        """
        Check if blocks for each component stay connected to each other
        This is used for multi-component goals - allows disconnection BETWEEN components
        """
        if not self.multi_component_goal or not self.allow_disconnection:
            return self.is_connected(state)
        
        # Convert state to list if needed
        state_list = list(state) if isinstance(state, frozenset) else state
        
        # Extract target blocks (exclude fixed blocks)
        target_blocks = [pos for pos in state_list if pos not in self.non_target_state]
        
        # CRITICAL FIX: Group blocks by their assigned component
        component_blocks = {}
        for pos in target_blocks:
            if pos in self.block_component_assignment:
                comp_idx = self.block_component_assignment[pos]
                if comp_idx not in component_blocks:
                    component_blocks[comp_idx] = []
                component_blocks[comp_idx].append(pos)
        
        # CRITICAL FIX: Check connectivity ONLY within each component (not between components)
        for comp_idx, blocks in component_blocks.items():
            if len(blocks) > 1 and not self.is_connected(blocks):
                return False
        
        # IMPORTANT: Return True even if components are disconnected from each other
        return True

    def get_articulation_points(self, state_set):
        """
        Find articulation points (critical points that if removed would disconnect the structure)
        Uses a modified DFS algorithm
        """
        state_hash = hash(frozenset(state_set))
        if state_hash in self.articulation_points_cache:
            return self.articulation_points_cache[state_hash]
            
        if len(state_set) <= 2:  # All points are critical in structures of size 1-2
            self.articulation_points_cache[state_hash] = set(state_set)
            return set(state_set)
            
        articulation_points = set()
        visited = set()
        discovery = {}
        low = {}
        parent = {}
        time = [0]  # Using list to allow modification inside nested function
        
        def dfs(u, time):
            cdef int children = 0
            visited.add(u)
            discovery[u] = low[u] = time[0]
            time[0] += 1
            
            # Visit all neighbors
            for dx, dy in self.directions:
                v = (u[0] + dx, u[1] + dy)
                if v in state_set:
                    if v not in visited:
                        children += 1
                        parent[v] = u
                        dfs(v, time)
                        
                        # Check if subtree rooted with v has a connection to ancestors of u
                        low[u] = min(low[u], low[v])
                        
                        # u is an articulation point if:
                        # 1) u is root and has two or more children
                        # 2) u is not root and low value of one of its children >= discovery value of u
                        if parent.get(u) is None and children > 1:
                            articulation_points.add(u)
                        if parent.get(u) is not None and low[v] >= discovery[u]:
                            articulation_points.add(u)
                            
                    elif v != parent.get(u):  # Update low value of u for parent function calls
                        low[u] = min(low[u], discovery[v])
        
        # Call DFS for all vertices
        for point in state_set:
            if point not in visited:
                dfs(point, time)
                
        self.articulation_points_cache[state_hash] = articulation_points
        return articulation_points
    
    def get_component_articulation_points(self, state):
        """
        Find articulation points for each component separately
        Used with multi-component goals
        """
        if not self.multi_component_goal or not self.allow_disconnection:
            return self.get_articulation_points(state)
            
        # Extract target blocks
        target_blocks = [pos for pos in state if pos not in self.non_target_state]
        
        # Group blocks by their assigned component
        component_blocks = {}
        for pos in target_blocks:
            if pos in self.block_component_assignment:
                comp_idx = self.block_component_assignment[pos]
                if comp_idx not in component_blocks:
                    component_blocks[comp_idx] = []
                component_blocks[comp_idx].append(pos)
        
        # Get articulation points for each component
        component_articulation_points = set()
        for comp_blocks in component_blocks.values():
            if len(comp_blocks) > 1:
                articulation_points = self.get_articulation_points(set(comp_blocks))
                component_articulation_points.update(articulation_points)
        
        return component_articulation_points
    
    def has_overlapping_blocks(self, state):
        """Check if a state has any overlapping blocks - enhanced for robustness"""
        # Convert to list if needed
        if isinstance(state, frozenset):
            state_list = list(state)
        else:
            state_list = state
            
        # Check for duplicates
        block_positions = set()
        for pos in state_list:
            if pos in block_positions:
                return True
            block_positions.add(pos)
            
        return False
        
    def get_valid_block_moves(self, state):
        """
        Generate valid moves for blocks
        A valid block move shifts the target elements in the same direction
        Fixed blocks remain stationary
        For multi-component goals, each component can move independently
        
        Enhanced with stricter overlap prevention and proper disconnection support
        """
        valid_moves = []
        
        # Skip states with overlapping blocks
        if self.has_overlapping_blocks(state):
            return valid_moves
            
        # Extract movable blocks (target blocks)
        state_list = list(state)
        
        # If there are fixed blocks, they should remain stationary
        if self.allow_disconnection and self.non_target_state:
            # Only move target blocks
            movable_blocks = [pos for pos in state_list if pos not in self.non_target_state]
            fixed_blocks = [pos for pos in state_list if pos in self.non_target_state]
        else:
            # Move all blocks
            movable_blocks = state_list
            fixed_blocks = []
        
        # For multi-component goals, allow component-by-component movement
        if self.allow_disconnection and self.multi_component_goal:
            # Group blocks by component
            component_blocks = {}
            for pos in movable_blocks:
                if pos in self.block_component_assignment:
                    comp_idx = self.block_component_assignment[pos]
                    if comp_idx not in component_blocks:
                        component_blocks[comp_idx] = []
                    component_blocks[comp_idx].append(pos)
            
            # Consider moves for each component independently
            for comp_idx, blocks in component_blocks.items():
                # Try moving this component in each direction
                for dx, dy in self.directions:
                    # Calculate new positions after moving this component
                    new_positions = [(pos[0] + dx, pos[1] + dy) for pos in blocks]
                    
                    # Check if all new positions are valid:
                    # 1. Within bounds
                    # 2. Not occupied by fixed blocks
                    # 3. Not occupied by blocks from other components
                    # 4. No position duplication (no overlap)
                    all_valid = True
                    new_pos_set = set()
                    
                    for pos in new_positions:
                        # Check bounds
                        if not (0 <= pos[0] < self.grid_size[0] and 0 <= pos[1] < self.grid_size[1]):
                            all_valid = False
                            break
                            
                        # Check collision with fixed blocks
                        if pos in fixed_blocks:
                            all_valid = False
                            break
                            
                        # Check collision with other components (which remain stationary)
                        other_component_blocks = []
                        for other_idx, other_blocks in component_blocks.items():
                            if other_idx != comp_idx:
                                other_component_blocks.extend(other_blocks)
                        
                        if pos in other_component_blocks:
                            all_valid = False
                            break
                            
                        # Check for duplicates (overlap)
                        if pos in new_pos_set:
                            all_valid = False
                            break
                            
                        new_pos_set.add(pos)
                    
                    # Only consider moves that keep all positions valid
                    if all_valid:
                        # Create new state with this component's blocks moved
                        # and all other blocks in their original positions
                        new_state = list(fixed_blocks)  # Start with fixed blocks
                        
                        # Add unmoved components
                        for other_idx, other_blocks in component_blocks.items():
                            if other_idx != comp_idx:
                                new_state.extend(other_blocks)
                        
                        # Add moved blocks from this component
                        new_state.extend(new_positions)
                        
                        # Perform a final overlap check before adding to valid moves
                        if not self.has_overlapping_blocks(new_state):
                            # CRITICAL FIX: Only check connectivity WITHIN this component, not between components
                            # This allows components to disconnect from each other
                            if len(new_positions) <= 1 or self.is_connected(new_positions):
                                valid_moves.append(frozenset(new_state))
        else:
            # Standard movement for all blocks together
            for dx, dy in self.directions:
                # Calculate new positions after moving
                new_positions = [(pos[0] + dx, pos[1] + dy) for pos in movable_blocks]
                
                # Check if all new positions are valid:
                # 1. Within bounds
                # 2. Not occupied by fixed blocks
                # 3. No position duplication (no overlap)
                all_valid = True
                new_pos_set = set()
                
                for pos in new_positions:
                    # Check bounds
                    if not (0 <= pos[0] < self.grid_size[0] and 0 <= pos[1] < self.grid_size[1]):
                        all_valid = False
                        break
                        
                    # Check collision with fixed blocks
                    if pos in fixed_blocks:
                        all_valid = False
                        break
                        
                    # Check for duplicates (overlap)
                    if pos in new_pos_set:
                        all_valid = False
                        break
                        
                    new_pos_set.add(pos)
                
                # Only consider moves that keep all positions valid
                if all_valid:
                    # Combine with fixed blocks to create the new state
                    new_state = frozenset(new_positions + fixed_blocks)
                    
                    # Final overlap check
                    if not self.has_overlapping_blocks(new_state):
                        # For connectivity check, adapt based on goal structure
                        if self.allow_disconnection:
                            # For single component with disconnection allowed
                            target_blocks = [pos for pos in new_state if pos not in self.non_target_state]
                            
                            # For multi-component, each component should be connected internally
                            if self.multi_component_goal:
                                if self.check_component_connectivity(new_state):
                                    valid_moves.append(new_state)
                            # For single-component goals, just check target connectivity
                            elif self.is_connected(target_blocks) or len(target_blocks) <= 1:
                                valid_moves.append(new_state)
                        else:
                            # Standard connectivity check for regular goals
                            if self.is_connected(new_state):
                                valid_moves.append(new_state)
        
        return valid_moves

    def get_valid_morphing_moves(self, state):
        """
        Generate valid morphing moves
        Supports multiple simultaneous block movements with minimum requirement
        Fixed blocks remain stationary, only target blocks move
        """
        state_key = hash(state)
        if state_key in self.valid_moves_cache:
            return self.valid_moves_cache[state_key]
            
        # Skip states with overlapping blocks
        if self.has_overlapping_blocks(state):
            self.valid_moves_cache[state_key] = []
            return []
            
        # Get single block moves first
        single_moves = []
        state_set = set(state)
        
        # Identify fixed blocks vs movable blocks
        fixed_blocks = set()
        if self.allow_disconnection:
            fixed_blocks = state_set.intersection(self.non_target_state)
        
        # If there are fixed blocks, they should remain stationary
        if self.allow_disconnection and fixed_blocks:
            # Only consider target blocks as movable
            movable_candidates = state_set - fixed_blocks
            
            if self.multi_component_goal:
                # For multi-component goals, handle components independently
                component_movable_points = {}
                
                # Group movable candidates by component
                for pos in movable_candidates:
                    if pos in self.block_component_assignment:
                        comp_idx = self.block_component_assignment[pos]
                        if comp_idx not in component_movable_points:
                            component_movable_points[comp_idx] = set()
                        
                        # Skip articulation points for each component
                        comp_blocks = [p for p in movable_candidates 
                                   if p in self.block_component_assignment and 
                                   self.block_component_assignment[p] == comp_idx]
                        
                        if len(comp_blocks) > 1:
                            comp_articulation_points = self.get_articulation_points(set(comp_blocks))
                            if pos not in comp_articulation_points:
                                component_movable_points[comp_idx].add(pos)
                            # Allow moving one articulation point if necessary
                            elif len(comp_articulation_points) == len(comp_blocks):
                                # Find a safe articulation point to move
                                for art_point in comp_articulation_points:
                                    temp_blocks = set(comp_blocks)
                                    temp_blocks.remove(art_point)
                                    if len(temp_blocks) <= 1 or self.is_connected(temp_blocks):
                                        component_movable_points[comp_idx].add(art_point)
                                        break
                        else:
                            # Singleton components can always move
                            component_movable_points[comp_idx].add(pos)
                
                # Combine all movable points from all components
                movable_points = set()
                for points in component_movable_points.values():
                    movable_points.update(points)
            else:
                # For single-component goals with disconnection
                target_blocks = list(movable_candidates)
                if target_blocks:
                    target_articulation_points = self.get_articulation_points(set(target_blocks))
                    movable_points = set(target_blocks) - target_articulation_points
                    
                    # If all target points are critical, try moving one anyway
                    if not movable_points and target_articulation_points and len(target_blocks) > 1:
                        for point in target_articulation_points:
                            # Try removing and see if structure remains connected
                            temp_target_blocks = set(target_blocks)
                            temp_target_blocks.remove(point)
                            if self.is_connected(temp_target_blocks) or len(temp_target_blocks) <= 1:
                                movable_points.add(point)
                                break
                else:
                    movable_points = set()
        else:
            # Standard connectivity rules apply to all blocks
            articulation_points = self.get_articulation_points(state_set)
            movable_points = state_set - articulation_points
            
            # If all points are critical, try moving one anyway
            if not movable_points and articulation_points and len(state_set) > 1:
                for point in articulation_points:
                    # Try removing and see if structure remains connected
                    temp_state = state_set.copy()
                    temp_state.remove(point)
                    if self.is_connected(temp_state) or len(temp_state) <= 1:
                        movable_points.add(point)
                        break
        
        # Generate single block moves
        for point in movable_points:
            # Skip fixed blocks
            if point in fixed_blocks:
                continue
                
            # Try moving in each direction
            for dx, dy in self.directions:
                new_pos = (point[0] + dx, point[1] + dy)
                
                # Skip if out of bounds
                if not (0 <= new_pos[0] < self.grid_size[0] and 
                        0 <= new_pos[1] < self.grid_size[1]):
                    continue
                
                # Skip if already occupied
                if new_pos in state_set:
                    continue
                
                # Create new state by moving the point
                new_state_set = state_set.copy()
                new_state_set.remove(point)
                new_state_set.add(new_pos)
                
                # Check for overlapping positions
                if len(new_state_set) != len(state_set):
                    continue
                
                if self.allow_disconnection:
                    # With disconnection allowed, we need to check:
                    # 1. No overlap between target and fixed blocks
                    new_target_blocks = [pos for pos in new_state_set if pos not in self.non_target_state]
                    new_fixed_blocks = [pos for pos in new_state_set if pos in self.non_target_state]
                    
                    # Skip if any target block occupies the same position as a fixed block
                    if any(pos in new_fixed_blocks for pos in new_target_blocks):
                        continue
                    
                    # 2. For multi-component goals, check connectivity within each component
                    if self.multi_component_goal:
                        # Check if connectivity is maintained within each component
                        if self.check_component_connectivity(new_state_set):
                            single_moves.append((point, new_pos))
                    else:
                        # For single component, just check target block connectivity
                        if self.is_connected(new_target_blocks) or len(new_target_blocks) <= 1:
                            single_moves.append((point, new_pos))
                else:
                    # Standard connectivity check for regular goals
                    has_adjacent = False
                    for adj_dx, adj_dy in self.directions:
                        adj_pos = (new_pos[0] + adj_dx, new_pos[1] + adj_dy)
                        if adj_pos in new_state_set and adj_pos != new_pos:
                            has_adjacent = True
                            break
                    
                    # Only consider moves that maintain connectivity
                    if has_adjacent and self.is_connected(new_state_set):
                        single_moves.append((point, new_pos))
                    
        # Start with empty valid moves list
        valid_moves = []
        
        # Generate multi-block moves
        for k in range(self.min_simultaneous_moves, min(self.max_simultaneous_moves + 1, len(single_moves) + 1)):
            # Generate combinations of k moves
            for combo in self._generate_move_combinations(single_moves, k):
                # Check if the combination is valid (no conflicts)
                if self._is_valid_move_combination(combo, state_set):
                    # Apply the combination
                    new_state = self._apply_moves(state_set, combo)
                    
                    # Check for overlapping positions
                    if self.has_overlapping_blocks(new_state):
                        continue
                    
                    # Additional validation for goal with fewer blocks
                    if self.allow_disconnection:
                        # Extract target and fixed blocks from the new state
                        new_target_blocks = [pos for pos in new_state if pos not in self.non_target_state]
                        new_fixed_blocks = [pos for pos in new_state if pos in self.non_target_state]
                        
                        # Skip if any target block occupies the same position as a fixed block
                        if any(pos in new_fixed_blocks for pos in new_target_blocks):
                            continue
                            
                        # For multi-component goals, check connectivity within each component
                        if self.multi_component_goal:
                            if self.check_component_connectivity(new_state):
                                valid_moves.append(frozenset(new_state))
                        else:
                            # Check if target blocks remain connected to each other
                            if self.is_connected(new_target_blocks) or len(new_target_blocks) <= 1:
                                valid_moves.append(frozenset(new_state))
                    else:
                        # Check full connectivity for standard goal
                        if self.is_connected(new_state):
                            valid_moves.append(frozenset(new_state))
        
        # If no valid moves with min_simultaneous_moves, fallback to single moves if allowed
        if not valid_moves and self.min_simultaneous_moves == 1:
            valid_moves = []
            for move in single_moves:
                new_state = self._apply_moves(state_set, [move])
                
                # Skip states with overlapping blocks
                if self.has_overlapping_blocks(new_state):
                    continue
                    
                # Additional validation for goal with fewer blocks
                if self.allow_disconnection:
                    # Extract target and fixed blocks from the new state
                    new_target_blocks = [pos for pos in new_state if pos not in self.non_target_state]
                    new_fixed_blocks = [pos for pos in new_state if pos in self.non_target_state]
                    
                    # Skip if any target block occupies the same position as a fixed block
                    if any(pos in new_fixed_blocks for pos in new_target_blocks):
                        continue
                    
                    # For multi-component goals, check connectivity within each component
                    if self.multi_component_goal:
                        if self.check_component_connectivity(new_state):
                            valid_moves.append(frozenset(new_state))
                    else:
                        # Check if target blocks remain connected to each other
                        if self.is_connected(new_target_blocks) or len(new_target_blocks) <= 1:
                            valid_moves.append(frozenset(new_state))
                else:
                    # Check full connectivity for standard goal
                    if self.is_connected(new_state):
                        valid_moves.append(frozenset(new_state))
        
        # Cache results
        self.valid_moves_cache[state_key] = valid_moves
        return valid_moves
    
    def _generate_move_combinations(self, single_moves, k):
        """Generate all combinations of k moves from the list of single moves"""
        if k == 1:
            return [[move] for move in single_moves]
        
        result = []
        for i in range(len(single_moves) - k + 1):
            move = single_moves[i]
            for combo in self._generate_move_combinations(single_moves[i+1:], k-1):
                result.append([move] + combo)
        
        return result
    
    def _is_valid_move_combination(self, moves, state_set):
        """Check if a combination of moves is valid (no conflicts)"""
        # Extract source and target positions
        sources = set()
        targets = set()
        
        for src, tgt in moves:
            # Check for overlapping sources or targets
            if src in sources or tgt in targets:
                return False
            sources.add(src)
            targets.add(tgt)
            
            # Check that no target is also a source for another move
            if tgt in sources or src in targets:
                return False
        
        # Additional check: Make sure targets don't collide with unmoved blocks
        # Identify blocks that won't be moving in this step
        remaining_blocks = state_set - sources
        
        # Check that no target position collides with a block that isn't moving
        for tgt in targets:
            if tgt in remaining_blocks:
                return False
        
        return True
    
    def _apply_moves(self, state_set, moves):
        """Apply a list of moves to the state"""
        new_state = state_set.copy()
        for src, tgt in moves:
            new_state.remove(src)
            new_state.add(tgt)
        return new_state
    
    def get_smart_chain_moves(self, state):
        """
        Generate chain moves where one block moves into the space of another
        while that block moves elsewhere
        Fixed blocks remain stationary
        Supports multi-component goals with independent component movement
        """
        cdef double min_dist
        cdef int dx, dy
        
        # Skip states with overlapping blocks
        if self.has_overlapping_blocks(state):
            return []
            
        state_set = set(state)
        valid_moves = []
        
        # Identify fixed blocks
        fixed_blocks = set()
        if self.allow_disconnection:
            fixed_blocks = state_set.intersection(self.non_target_state)
        
        # For multi-component goals, we'll process each component separately
        if self.allow_disconnection and self.multi_component_goal:
            # Extract target blocks
            target_blocks = [pos for pos in state_set if pos not in fixed_blocks]
            
            # Group by component
            component_blocks = {}
            for pos in target_blocks:
                if pos in self.block_component_assignment:
                    comp_idx = self.block_component_assignment[pos]
                    if comp_idx not in component_blocks:
                        component_blocks[comp_idx] = []
                    component_blocks[comp_idx].append(pos)
                    
            # Process each component independently
            for comp_idx, blocks in component_blocks.items():
                # Get the goal positions for this component
                goal_component = self.goal_components[comp_idx] if comp_idx < len(self.goal_components) else []
                
                # For each block in this component, try to move it toward its goal
                for pos in blocks:
                    # Determine goal position based on component assignment
                    closest_goal = None
                    min_dist = float('inf')
                    
                    # Find closest unoccupied position in this component's goal
                    for goal_pos in goal_component:
                        if goal_pos not in state_set:  # Only consider unoccupied goals
                            dist = abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])
                            if dist < min_dist:
                                min_dist = dist
                                closest_goal = goal_pos
                    
                    if not closest_goal:
                        continue
                        
                    # Calculate direction toward goal
                    dx = 1 if closest_goal[0] > pos[0] else -1 if closest_goal[0] < pos[0] else 0
                    dy = 1 if closest_goal[1] > pos[1] else -1 if closest_goal[1] < pos[1] else 0
                    
                    # Try moving in that direction
                    next_pos = (pos[0] + dx, pos[1] + dy)
                    
                    # Skip if out of bounds
                    if not (0 <= next_pos[0] < self.grid_size[0] and 
                            0 <= next_pos[1] < self.grid_size[1]):
                        continue
                    
                    # If next position is occupied, try chain move
                    if next_pos in state_set:
                        # Skip if the occupied position is a fixed block
                        if next_pos in fixed_blocks:
                            continue
                            
                        # Determine which component the blocking block belongs to
                        blocking_comp_idx = None
                        if next_pos in self.block_component_assignment:
                            blocking_comp_idx = self.block_component_assignment[next_pos]
                        
                        # Try moving the blocking block elsewhere
                        for chain_dx, chain_dy in self.directions:
                            chain_pos = (next_pos[0] + chain_dx, next_pos[1] + chain_dy)
                            
                            # Skip if out of bounds, occupied, or original position
                            if not (0 <= chain_pos[0] < self.grid_size[0] and 
                                    0 <= chain_pos[1] < self.grid_size[1]):
                                continue
                            if chain_pos in state_set or chain_pos == pos:
                                continue
                            
                            # Create new state by moving both blocks
                            new_state_set = state_set.copy()
                            new_state_set.remove(pos)
                            new_state_set.remove(next_pos)
                            new_state_set.add(next_pos)
                            new_state_set.add(chain_pos)
                            
                            # Check for overlapping positions
                            if len(new_state_set) != len(state_set):
                                continue
                            
                            # Check connectivity within each component
                            component_connectivity_maintained = True
                            
                            # Check connectivity for the component of the moved block
                            moved_comp_blocks = [p for p in new_state_set 
                                               if p not in fixed_blocks and 
                                               p in self.block_component_assignment and
                                               self.block_component_assignment[p] == comp_idx]
                            
                            if len(moved_comp_blocks) > 1 and not self.is_connected(moved_comp_blocks):
                                component_connectivity_maintained = False
                            
                            # If blocking block is from another component, check that too
                            if blocking_comp_idx is not None and blocking_comp_idx != comp_idx:
                                blocked_comp_blocks = [p for p in new_state_set 
                                                    if p not in fixed_blocks and 
                                                    p in self.block_component_assignment and
                                                    self.block_component_assignment[p] == blocking_comp_idx]
                                
                                if len(blocked_comp_blocks) > 1 and not self.is_connected(blocked_comp_blocks):
                                    component_connectivity_maintained = False
                            
                            if component_connectivity_maintained:
                                valid_moves.append(frozenset(new_state_set))
                    
                    # If next position is unoccupied, try direct move
                    else:
                        new_state_set = state_set.copy()
                        new_state_set.remove(pos)
                        new_state_set.add(next_pos)
                        
                        # Check for overlapping positions
                        if len(new_state_set) != len(state_set):
                            continue
                        
                        # Check connectivity within this component
                        comp_blocks = [p for p in new_state_set 
                                     if p not in fixed_blocks and 
                                     p in self.block_component_assignment and
                                     self.block_component_assignment[p] == comp_idx]
                        
                        if len(comp_blocks) <= 1 or self.is_connected(comp_blocks):
                            valid_moves.append(frozenset(new_state_set))
        else:
            # Original implementation for single component goals
            # For each movable block, try to move it toward a goal position
            for pos in state_set:
                # Skip fixed blocks
                if pos in fixed_blocks:
                    continue
                    
                # Determine goal position based on component assignment
                closest_goal = None
                min_dist = float('inf')
                
                if self.multi_component_goal and pos in self.block_component_assignment:
                    # Get the component this block is assigned to
                    comp_idx = self.block_component_assignment[pos]
                    comp_positions = self.goal_components[comp_idx]
                    
                    # Find closest unoccupied position in this component
                    for goal_pos in comp_positions:
                        if goal_pos not in state_set:  # Only consider unoccupied goals
                            dist = abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])
                            if dist < min_dist:
                                min_dist = dist
                                closest_goal = goal_pos
                else:
                    # Standard closest goal search
                    for goal_pos in self.goal_state:
                        if goal_pos not in state_set:  # Only consider unoccupied goals
                            dist = abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])
                            if dist < min_dist:
                                min_dist = dist
                                closest_goal = goal_pos
                
                if not closest_goal:
                    continue
                    
                # Calculate direction toward goal
                dx = 1 if closest_goal[0] > pos[0] else -1 if closest_goal[0] < pos[0] else 0
                dy = 1 if closest_goal[1] > pos[1] else -1 if closest_goal[1] < pos[1] else 0
                
                # Try moving in that direction
                next_pos = (pos[0] + dx, pos[1] + dy)
                
                # Skip if out of bounds
                if not (0 <= next_pos[0] < self.grid_size[0] and 
                        0 <= next_pos[1] < self.grid_size[1]):
                    continue
                
                # If next position is occupied, try chain move
                if next_pos in state_set:
                    # Skip if the occupied position is a fixed block
                    if next_pos in fixed_blocks:
                        continue
                        
                    # Try moving the blocking block elsewhere
                    for chain_dx, chain_dy in self.directions:
                        chain_pos = (next_pos[0] + chain_dx, next_pos[1] + chain_dy)
                        
                        # Skip if out of bounds, occupied, or original position
                        if not (0 <= chain_pos[0] < self.grid_size[0] and 
                                0 <= chain_pos[1] < self.grid_size[1]):
                            continue
                        if chain_pos in state_set or chain_pos == pos:
                            continue
                        
                        # Create new state by moving both blocks
                        new_state_set = state_set.copy()
                        new_state_set.remove(pos)
                        new_state_set.remove(next_pos)
                        new_state_set.add(next_pos)
                        new_state_set.add(chain_pos)
                        
                        # Check for overlapping positions
                        if len(new_state_set) != len(state_set):
                            continue
                        
                        # Handle connectivity based on goal type
                        if self.allow_disconnection:
                            # Extract target and fixed blocks
                            new_target_blocks = [p for p in new_state_set if p not in self.non_target_state]
                            new_fixed_blocks = [p for p in new_state_set if p in self.non_target_state]
                            
                            # Skip if any target block occupies the same position as a fixed block
                            if any(p in new_fixed_blocks for p in new_target_blocks):
                                continue
                                
                            # For multi-component goals, check connectivity within each component
                            if self.multi_component_goal:
                                if self.check_component_connectivity(new_state_set):
                                    valid_moves.append(frozenset(new_state_set))
                            else:
                                # Check target block connectivity
                                if self.is_connected(new_target_blocks) or len(new_target_blocks) <= 1:
                                    valid_moves.append(frozenset(new_state_set))
                        else:
                            # Standard connectivity for all blocks
                            if self.is_connected(new_state_set):
                                valid_moves.append(frozenset(new_state_set))
                
                # If next position is unoccupied, try direct move
                else:
                    new_state_set = state_set.copy()
                    new_state_set.remove(pos)
                    new_state_set.add(next_pos)
                    
                    # Check for overlapping positions
                    if len(new_state_set) != len(state_set):
                        continue
                    
                    # Handle connectivity based on goal type
                    if self.allow_disconnection:
                        # Extract target and fixed blocks
                        new_target_blocks = [p for p in new_state_set if p not in self.non_target_state]
                        new_fixed_blocks = [p for p in new_state_set if p in self.non_target_state]
                        
                        # Skip if any target block occupies the same position as a fixed block
                        if any(p in new_fixed_blocks for p in new_target_blocks):
                            continue
                            
                        # For multi-component goals, check connectivity within each component
                        if self.multi_component_goal:
                            if self.check_component_connectivity(new_state_set):
                                valid_moves.append(frozenset(new_state_set))
                        else:
                            # Check target block connectivity
                            if self.is_connected(new_target_blocks) or len(new_target_blocks) <= 1:
                                valid_moves.append(frozenset(new_state_set))
                    else:
                        # Standard connectivity for all blocks
                        if self.is_connected(new_state_set):
                            valid_moves.append(frozenset(new_state_set))
        
        return valid_moves

    def get_sliding_chain_moves(self, state):
        """
        Generate sliding chain moves where multiple blocks move in sequence
        to navigate tight spaces
        Fixed blocks remain stationary
        """
        # Skip states with overlapping blocks
        if self.has_overlapping_blocks(state):
            return []
            
        state_set = set(state)
        valid_moves = []
        
        # Identify fixed blocks
        fixed_blocks = set()
        if self.allow_disconnection:
            fixed_blocks = state_set.intersection(self.non_target_state)
        
        # For each movable block, try to initiate a sliding chain
        for pos in state_set:
            # Skip fixed blocks
            if pos in fixed_blocks:
                continue
                
            # Skip if it's a critical articulation point (unless for multi-component goal)
            if not self.allow_disconnection:
                articulation_points = self.get_articulation_points(state_set)
                if pos in articulation_points and len(articulation_points) <= 20:
                    continue
            elif self.allow_disconnection:
                if self.multi_component_goal:
                    # For multi-component goals, check if it's an articulation point in its component
                    if pos in self.block_component_assignment:
                        comp_idx = self.block_component_assignment[pos]
                        comp_blocks = [p for p in state_set if p not in fixed_blocks and 
                                    p in self.block_component_assignment and
                                    self.block_component_assignment[p] == comp_idx]
                        
                        if len(comp_blocks) > 1:
                            comp_articulation_points = self.get_articulation_points(set(comp_blocks))
                            if pos in comp_articulation_points and len(comp_articulation_points) <= 15:
                                continue
                else:
                    # For single component with disconnection
                    target_blocks = [p for p in state_set if p not in fixed_blocks]
                    if target_blocks:
                        target_articulation_points = self.get_articulation_points(set(target_blocks))
                        if pos in target_articulation_points and len(target_articulation_points) <= 15:
                            continue
                
            # Try sliding in each direction
            for dx, dy in self.directions:
                # Only consider diagonal moves for sliding chains
                if dx != 0 and dy != 0:
                    # Define the sliding path (up to 3 steps)
                    path = []
                    current_pos = pos
                    for _ in range(20):  # Maximum chain length
                        next_pos = (current_pos[0] + dx, current_pos[1] + dy)
                        # Stop if out of bounds
                        if not (0 <= next_pos[0] < self.grid_size[0] and 
                                0 <= next_pos[1] < self.grid_size[1]):
                            break
                        path.append(next_pos)
                        current_pos = next_pos
                    
                    # Try sliding the block along the path
                    for i, target_pos in enumerate(path):
                        # Skip if target is occupied
                        if target_pos in state_set:
                            continue
                            
                        # Create new state by moving the block
                        new_state_set = state_set.copy()
                        new_state_set.remove(pos)
                        new_state_set.add(target_pos)
                        
                        # Check for overlapping positions
                        if len(new_state_set) != len(state_set):
                            continue
                        
                        # Handle connectivity based on goal type
                        if self.allow_disconnection:
                            # Extract target and fixed blocks
                            new_target_blocks = [p for p in new_state_set if p not in self.non_target_state]
                            new_fixed_blocks = [p for p in new_state_set if p in self.non_target_state]
                            
                            # Skip if any target block occupies the same position as a fixed block
                            if any(p in new_fixed_blocks for p in new_target_blocks):
                                continue
                                
                            # For multi-component goals, check connectivity within each component
                            if self.multi_component_goal:
                                if self.check_component_connectivity(new_state_set):
                                    valid_moves.append(frozenset(new_state_set))
                            else:
                                # Check target block connectivity
                                if self.is_connected(new_target_blocks) or len(new_target_blocks) <= 1:
                                    valid_moves.append(frozenset(new_state_set))
                        else:
                            # Standard connectivity for all blocks
                            if self.is_connected(new_state_set):
                                valid_moves.append(frozenset(new_state_set))
                        
                        # No need to continue if we can't reach this position
                        break
        
        return valid_moves
    
    def get_all_valid_moves(self, state):
        """
        Combine all move generation methods to maximize options
        Allow proper disconnection for multi-component goals
        """
        # Skip states with overlapping blocks
        if self.has_overlapping_blocks(state):
            return []
            
        # Start with basic morphing moves
        basic_moves = self.get_valid_morphing_moves(state)
        
        # Add chain moves
        chain_moves = self.get_smart_chain_moves(state)
        
        # Add sliding chain moves
        sliding_moves = self.get_sliding_chain_moves(state)
        
        # CRITICAL FIX: For multi-component goals, add "disconnection moves"
        if self.allow_disconnection and self.multi_component_goal:
            disconnection_moves = self._generate_component_separation_moves(state)
            all_moves = list(set(basic_moves + chain_moves + sliding_moves + disconnection_moves))
        else:
            # Regular combined moves
            all_moves = list(set(basic_moves + chain_moves + sliding_moves))
        
        # Final check for overlapping blocks
        all_moves = [move for move in all_moves if not self.has_overlapping_blocks(move)]
        
        return all_moves
    
    def block_heuristic(self, state):
        """
        Heuristic for block movement phase:
        Calculate Manhattan distance from current target blocks to goal centroid
        For multi-component goals, sum distances to appropriate component centroids
        """
        if not state:
            return float('inf')
        
        # Skip states with overlapping blocks
        if self.has_overlapping_blocks(state):
            return float('inf')
            
        if self.allow_disconnection:
            # Extract target blocks from current state
            target_blocks = [pos for pos in state if pos not in self.non_target_state]
            if not target_blocks:
                return float('inf')
                
            if self.multi_component_goal:
                # Calculate distance for each component to its centroid
                total_distance = 0
                
                # Group blocks by their assigned component
                component_blocks = {}
                for pos in target_blocks:
                    if pos in self.block_component_assignment:
                        comp_idx = self.block_component_assignment[pos]
                        if comp_idx not in component_blocks:
                            component_blocks[comp_idx] = []
                        component_blocks[comp_idx].append(pos)
                
                # Calculate distance for each component
                for comp_idx, blocks in component_blocks.items():
                    if blocks:
                        comp_centroid = self.calculate_centroid(blocks)
                        goal_centroid = self.goal_component_centroids[comp_idx]
                        total_distance += (abs(comp_centroid[0] - goal_centroid[0]) + 
                                        abs(comp_centroid[1] - goal_centroid[1]))
                
                return total_distance
            else:
                # Standard centroid calculation for target blocks
                target_centroid = self.calculate_centroid(target_blocks)
                
                # Pure Manhattan distance between centroids
                return abs(target_centroid[0] - self.goal_centroid[0]) + abs(target_centroid[1] - self.goal_centroid[1])
        else:
            # Standard centroid calculation for all blocks
            current_centroid = self.calculate_centroid(state)
            
            # Pure Manhattan distance between centroids
            return abs(current_centroid[0] - self.goal_centroid[0]) + abs(current_centroid[1] - self.goal_centroid[1])

    def block_movement_phase(self, double time_limit=15):
        """
        Phase 1: Move blocks toward the goal centroid or component centroids
        If goal has fewer blocks than start, only move the target blocks
        Fixed blocks remain stationary throughout
        For multi-component goals, allow gradual separation
        """
        cdef double start_time
        cdef double min_distance = 1.0
        cdef double max_distance = 1.0
        cdef double centroid_distance, neighbor_distance, distance_penalty
        cdef double adjusted_heuristic, f_score, best_distance_diff, distance, distance_diff, best_distance
        cdef int tentative_g
        
        print("Starting Block Movement Phase...")
        if self.allow_disconnection:
            if self.multi_component_goal:
                print(f"Moving {len(self.target_block_list)} target blocks towards {len(self.goal_components)} component centroids")
                print(f"Components will be allowed to separate during movement")
            else:
                print(f"Moving only {len(self.target_block_list)} target blocks, keeping {len(self.fixed_block_list)} blocks stationary")
        
        start_time = time.time()

        # Create initial state - if disconnection is allowed, we need to include fixed blocks
        initial_state = self.start_state

        # Initialize A* search
        open_set = [(self.block_heuristic(initial_state), 0, initial_state)]
        closed_set = set()

        # Track path and g-scores
        g_score = {initial_state: 0}
        came_from = {initial_state: None}

        while open_set and time.time() - start_time < time_limit:
            # Get state with lowest f-score
            f, g, current = heapq.heappop(open_set)
    
            # Skip if already processed
            if current in closed_set:
                continue
            
            # Skip states with overlapping blocks
            if self.has_overlapping_blocks(current):
                continue
            
            # For multi-component goals, check if blocks are close enough to their component centroids
            if self.allow_disconnection and self.multi_component_goal:
                # Extract target blocks
                target_blocks = [pos for pos in current if pos not in self.non_target_state]
                
                # Group blocks by their assigned component
                component_blocks = {}
                for pos in target_blocks:
                    if pos in self.block_component_assignment:
                        comp_idx = self.block_component_assignment[pos]
                        if comp_idx not in component_blocks:
                            component_blocks[comp_idx] = []
                        component_blocks[comp_idx].append(pos)
                
                # Check if all components are close enough to their centroids
                all_components_close = True
                
                for comp_idx, blocks in component_blocks.items():
                    if blocks:
                        comp_centroid = self.calculate_centroid(blocks)
                        goal_centroid = self.goal_component_centroids[comp_idx]
                        component_distance = (abs(comp_centroid[0] - goal_centroid[0]) + 
                                           abs(comp_centroid[1] - goal_centroid[1]))
                        
                        # If any component is too far, continue movement
                        if component_distance > max_distance:
                            all_components_close = False
                            break
                
                if all_components_close:
                    print(f"All components reached appropriate distances from their target centroids")
                    return self.reconstruct_path(came_from, current)
            
            # For single-component goals, check distance to centroid as before
            elif self.allow_disconnection:
                # Extract target blocks
                target_blocks = [pos for pos in current if pos not in self.non_target_state]
                
                # Calculate centroid of target blocks only
                if target_blocks:
                    target_centroid = self.calculate_centroid(target_blocks)
                    centroid_distance = (abs(target_centroid[0] - self.goal_centroid[0]) + 
                                        abs(target_centroid[1] - self.goal_centroid[1]))
                else:
                    centroid_distance = float('inf')
                        
                if min_distance <= centroid_distance <= max_distance:
                    print(f"Blocks stopped 1 grid cell before goal centroid. Distance: {centroid_distance}")
                    return self.reconstruct_path(came_from, current)
            else:
                # Standard centroid calculation for all blocks
                current_centroid = self.calculate_centroid(current)
                centroid_distance = (abs(current_centroid[0] - self.goal_centroid[0]) + 
                                    abs(current_centroid[1] - self.goal_centroid[1]))
                        
                if min_distance <= centroid_distance <= max_distance:
                    print(f"Blocks stopped 1 grid cell before goal centroid. Distance: {centroid_distance}")
                    return self.reconstruct_path(came_from, current)
        
            closed_set.add(current)
    
            # Process neighbor states (block moves)
            for neighbor in self.get_valid_block_moves(current):
                # Skip states with overlapping blocks
                if self.has_overlapping_blocks(neighbor):
                    continue
                    
                if neighbor in closed_set:
                    continue
                
                # Calculate tentative g-score
                tentative_g = g_score[current] + 1
        
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # This is a better path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                
                    # Adjusted heuristic based on goal structure
                    if self.allow_disconnection and self.multi_component_goal:
                        # For multi-component goals, use the block_heuristic directly
                        # which handles component-specific distances
                        adjusted_heuristic = self.block_heuristic(neighbor)
                        
                        # Give bonus for states where components are appropriately separated
                        # This encourages components to move toward their separate goal locations
                        separated_components = True
                        target_blocks = [pos for pos in neighbor if pos not in self.non_target_state]
                        
                        # Group blocks by their assigned component
                        component_blocks = {}
                        for pos in target_blocks:
                            if pos in self.block_component_assignment:
                                comp_idx = self.block_component_assignment[pos]
                                if comp_idx not in component_blocks:
                                    component_blocks[comp_idx] = []
                                component_blocks[comp_idx].append(pos)
                        
                        # Check if components are properly separated
                        # This gives a bonus to states where the components are forming in appropriate locations
                        if self.check_component_connectivity(neighbor):
                            adjusted_heuristic *= 0.95  # Small bonus to encourage appropriate separation
                    elif self.allow_disconnection:
                        # For single-component goals with disconnection
                        neighbor_target_blocks = [pos for pos in neighbor if pos not in self.non_target_state]
                        if neighbor_target_blocks:
                            neighbor_centroid = self.calculate_centroid(neighbor_target_blocks)
                            neighbor_distance = (abs(neighbor_centroid[0] - self.goal_centroid[0]) + 
                                               abs(neighbor_centroid[1] - self.goal_centroid[1]))
                            
                            # Penalize distances that are too small
                            distance_penalty = 0
                            if neighbor_distance < min_distance:
                                distance_penalty = 10 * (min_distance - neighbor_distance)
                            
                            adjusted_heuristic = self.block_heuristic(neighbor) + distance_penalty
                        else:
                            adjusted_heuristic = float('inf')
                    else:
                        # Standard centroid calculation for regular goals
                        neighbor_centroid = self.calculate_centroid(neighbor)
                        neighbor_distance = (abs(neighbor_centroid[0] - self.goal_centroid[0]) + 
                                           abs(neighbor_centroid[1] - self.goal_centroid[1]))
                
                        # Penalize distances that are too small
                        distance_penalty = 0
                        if neighbor_distance < min_distance:
                            distance_penalty = 10 * (min_distance - neighbor_distance)
                    
                        adjusted_heuristic = self.block_heuristic(neighbor) + distance_penalty
                
                    f_score = tentative_g + adjusted_heuristic
            
                    # Add to open set
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        # If we exit the loop, either no path was found or time limit reached
        if time.time() - start_time >= time_limit:
            print("Block movement phase timed out!")
    
        # Return the best state we found
        if came_from:
            # Find state with appropriate distance to centroid
            best_state = None
            best_distance_diff = float('inf')
            
            for state in came_from.keys():
                # Skip states with overlapping blocks
                if self.has_overlapping_blocks(state):
                    continue
                
                # Handle multi-component goals
                if self.allow_disconnection and self.multi_component_goal:
                    # Extract target blocks
                    target_blocks = [pos for pos in state if pos not in self.non_target_state]
                    
                    # Group blocks by their assigned component
                    component_blocks = {}
                    for pos in target_blocks:
                        if pos in self.block_component_assignment:
                            comp_idx = self.block_component_assignment[pos]
                            if comp_idx not in component_blocks:
                                component_blocks[comp_idx] = []
                            component_blocks[comp_idx].append(pos)
                    
                    # Calculate total distance difference across all components
                    total_distance_diff = 0
                    all_within_range = True
                    
                    for comp_idx, blocks in component_blocks.items():
                        if blocks:
                            comp_centroid = self.calculate_centroid(blocks)
                            goal_centroid = self.goal_component_centroids[comp_idx]
                            component_distance = (abs(comp_centroid[0] - goal_centroid[0]) + 
                                               abs(comp_centroid[1] - goal_centroid[1]))
                            
                            # Calculate difference from acceptable range
                            if component_distance < min_distance:
                                total_distance_diff += min_distance - component_distance
                                all_within_range = False
                            elif component_distance > max_distance:
                                total_distance_diff += component_distance - max_distance
                                all_within_range = False
                    
                    # If all components are within range, this is the best state
                    if all_within_range:
                        best_state = state
                        break
                    
                    # Otherwise, track the state with minimal deviation
                    if total_distance_diff < best_distance_diff:
                        best_distance_diff = total_distance_diff
                        best_state = state
                
                # For single-component goals or standard goals
                else:
                    if self.allow_disconnection:
                        # Calculate distance for target blocks only
                        target_blocks = [pos for pos in state if pos not in self.non_target_state]
                        if target_blocks:
                            state_centroid = self.calculate_centroid(target_blocks)
                            distance = (abs(state_centroid[0] - self.goal_centroid[0]) + 
                                      abs(state_centroid[1] - self.goal_centroid[1]))
                        else:
                            distance = float('inf')
                    else:
                        # Standard centroid calculation
                        state_centroid = self.calculate_centroid(state)
                        distance = (abs(state_centroid[0] - self.goal_centroid[0]) + 
                                  abs(state_centroid[1] - self.goal_centroid[1]))
                
                    # We want a state that's as close as possible to our target distance range
                    if distance < min_distance:
                        distance_diff = min_distance - distance
                    elif distance > max_distance:
                        distance_diff = distance - max_distance
                    else:
                        # Distance is within our desired range
                        best_state = state
                        break
                    
                    if distance_diff < best_distance_diff:
                        best_distance_diff = distance_diff
                        best_state = state
            
            if best_state:
                if self.allow_disconnection and self.multi_component_goal:
                    # Extract target blocks
                    target_blocks = [pos for pos in best_state if pos not in self.non_target_state]
                    
                    # Group blocks by their assigned component
                    component_blocks = {}
                    for pos in target_blocks:
                        if pos in self.block_component_assignment:
                            comp_idx = self.block_component_assignment[pos]
                            if comp_idx not in component_blocks:
                                component_blocks[comp_idx] = []
                            component_blocks[comp_idx].append(pos)
                    
                    # Print distance for each component
                    for comp_idx, blocks in component_blocks.items():
                        if blocks:
                            comp_centroid = self.calculate_centroid(blocks)
                            goal_centroid = self.goal_component_centroids[comp_idx]
                            component_distance = (abs(comp_centroid[0] - goal_centroid[0]) + 
                                               abs(comp_centroid[1] - goal_centroid[1]))
                            print(f"Component {comp_idx+1} reached centroid distance: {component_distance:.2f}")
                
                elif self.allow_disconnection:
                    # Calculate distance for target blocks only
                    target_blocks = [pos for pos in best_state if pos not in self.non_target_state]
                    if target_blocks:
                        best_centroid = self.calculate_centroid(target_blocks)
                        best_distance = (abs(best_centroid[0] - self.goal_centroid[0]) + 
                                       abs(best_centroid[1] - self.goal_centroid[1]))
                        print(f"Best block position found with centroid distance: {best_distance}")
                else:
                    # Standard centroid calculation
                    best_centroid = self.calculate_centroid(best_state)
                    best_distance = (abs(best_centroid[0] - self.goal_centroid[0]) + 
                                   abs(best_centroid[1] - self.goal_centroid[1]))
                    print(f"Best block position found with centroid distance: {best_distance}")
                
                return self.reconstruct_path(came_from, best_state)
        
        return [self.start_state]  # No movement possible

    def improved_morphing_heuristic(self, state):
        """
        Improved heuristic for morphing phase:
        Uses bipartite matching to find optimal assignment of blocks to goal positions
        Handles multi-component goals by matching within each component
        """
        cdef double total_distance = 0
        cdef double min_dist
        cdef int best_j
        cdef int matching_positions
        cdef double connectivity_bonus
        
        if not state:
            return float('inf')
            
        # Skip states with overlapping blocks
        if self.has_overlapping_blocks(state):
            return float('inf')
        
        # If we're targeting a subset of blocks (goal has fewer blocks)
        if self.allow_disconnection:
            # Extract target blocks (exclude fixed blocks)
            target_blocks = [pos for pos in state if pos not in self.non_target_state]
            
            # If number of target blocks doesn't match goal positions, this shouldn't happen
            if len(target_blocks) != len(self.goal_positions):
                return float('inf')
                
            if self.multi_component_goal:
                # For multi-component goals, calculate distance for each component separately
                
                # Group blocks and goal positions by component
                component_blocks = {}
                component_goals = {}
                
                for pos in target_blocks:
                    if pos in self.block_component_assignment:
                        comp_idx = self.block_component_assignment[pos]
                        if comp_idx not in component_blocks:
                            component_blocks[comp_idx] = []
                        component_blocks[comp_idx].append(pos)
                
                for i, component in enumerate(self.goal_components):
                    component_goals[i] = component
                
                # Calculate distance for each component
                for comp_idx, blocks in component_blocks.items():
                    goals = component_goals.get(comp_idx, [])
                    
                    # Skip if no blocks or goals for this component
                    if not blocks or not goals:
                        continue
                    
                    # If block count doesn't match goal count for this component
                    if len(blocks) != len(goals):
                        return float('inf')
                    
                    # Build distance matrix for this component
                    distance_matrix = []
                    for pos in blocks:
                        row = []
                        for goal_pos in goals:
                            # Manhattan distance
                            dist = abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])
                            row.append(dist)
                        distance_matrix.append(row)
                    
                    # Greedy assignment for this component
                    assigned_cols = set()
                    
                    for i in range(len(blocks)):
                        # Find closest unassigned goal position
                        min_dist = float('inf')
                        best_j = -1
                        
                        for j in range(len(goals)):
                            if j not in assigned_cols and distance_matrix[i][j] < min_dist:
                                min_dist = distance_matrix[i][j]
                                best_j = j
                        
                        if best_j != -1:
                            assigned_cols.add(best_j)
                            total_distance += min_dist
                        else:
                            # No assignment possible
                            return float('inf')
                    
                    # Add bonus for blocks already in correct positions
                    matching_positions = sum(1 for pos in blocks if pos in goals)
                    total_distance -= matching_positions * 0.5  # Bonus to encourage matches
            else:
                # Standard assignment for single component goal
                goal_list = list(self.goal_state)
                
                # Build distance matrix for all target blocks
                distance_matrix = []
                for pos in target_blocks:
                    row = []
                    for goal_pos in goal_list:
                        # Manhattan distance
                        dist = abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])
                        row.append(dist)
                    distance_matrix.append(row)
                
                # Greedy assignment
                assigned_cols = set()
                
                for i in range(len(target_blocks)):
                    # Find closest unassigned goal position
                    min_dist = float('inf')
                    best_j = -1
                    
                    for j in range(len(goal_list)):
                        if j not in assigned_cols and distance_matrix[i][j] < min_dist:
                            min_dist = distance_matrix[i][j]
                            best_j = j
                    
                    if best_j != -1:
                        assigned_cols.add(best_j)
                        total_distance += min_dist
                    else:
                        # No assignment possible
                        return float('inf')
                
                # Add connectivity bonus: prefer states that have more blocks in goal positions
                matching_positions = sum(1 for pos in target_blocks if pos in self.goal_state)
                connectivity_bonus = -matching_positions * 0.5  # Negative to encourage more matches
                
                total_distance += connectivity_bonus
        else:
            # Original logic for when goal has same number of blocks as start
            state_list = list(state)
            goal_list = list(self.goal_state)
            
            # Build distance matrix
            distances = []
            for pos in state_list:
                row = []
                for goal_pos in goal_list:
                    # Manhattan distance
                    dist = abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])
                    row.append(dist)
                distances.append(row)
            
            # Use greedy assignment algorithm
            assigned_cols = set()
            
            # Sort rows by minimum distance
            row_indices = list(range(len(state_list)))
            row_indices.sort(key=lambda i: min(distances[i]))
            
            for i in row_indices:
                # Find closest unassigned goal position
                min_dist = float('inf')
                best_j = -1
                
                for j in range(len(goal_list)):
                    if j not in assigned_cols and distances[i][j] < min_dist:
                        min_dist = distances[i][j]
                        best_j = j
                
                if best_j != -1:
                    assigned_cols.add(best_j)
                    total_distance += min_dist
                else:
                    # No assignment possible
                    return float('inf')
            
            # Add connectivity bonus: prefer states that have more blocks in goal positions
            matching_positions = len(state.intersection(self.goal_state))
            connectivity_bonus = -matching_positions * 0.5  # Negative to encourage more matches
            
            total_distance += connectivity_bonus
            
        return total_distance
    
# New method to search for a single component
    def search_single_component(self, comp_idx, start_state, target_blocks, time_limit=15):
        """Search for a single component"""
        print(f"Starting search for Component {comp_idx+1} with {len(target_blocks)} blocks")
        
        # Create goal state for this component
        component_goal = self.goal_components[comp_idx]
        
        # Direct implementation rather than creating a new agent
        # (a) Block movement phase
        block_time_limit = time_limit * 0.3
        
        # First, get blocks to the vicinity of the goal component
        component_centroid = self.goal_component_centroids[comp_idx]
        
        # Initial state includes all blocks
        initial_state = start_state
        g_score = {initial_state: 0}
        came_from = {initial_state: None}
        
        # Simple distance heuristic for this component
        def component_heuristic(state):
            # Extract target blocks for this component
            comp_blocks = [pos for pos in state 
                        if pos in self.block_component_assignment and 
                        self.block_component_assignment[pos] == comp_idx]
            
            if not comp_blocks:
                return float('inf')
            
            # Calculate centroid distance
            comp_centroid = self.calculate_centroid(comp_blocks)
            return abs(comp_centroid[0] - component_centroid[0]) + abs(comp_centroid[1] - component_centroid[1])
        
        # Run simple A* search to get blocks near goal
        open_set = [(component_heuristic(initial_state), 0, initial_state)]
        start_time = time.time()
        
        while open_set and time.time() - start_time < block_time_limit:
            f, g, current = heapq.heappop(open_set)
            
            # Check if close enough to goal
            if component_heuristic(current) <= 2.0:
                break
                
            # Generate neighbors
            for neighbor in self.get_valid_block_moves(current):
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    f_score = tentative_g + component_heuristic(neighbor)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
        
        # Get the best state
        block_final_state = min(g_score.keys(), key=component_heuristic)
        block_path = self.reconstruct_path(came_from, block_final_state)
        
        # (b) Morphing phase
        morphing_time_limit = time_limit * 0.7
        morphing_path = self.smarter_morphing_phase(frozenset(block_final_state), morphing_time_limit)
        
        # Combine paths
        combined_path = block_path[:-1] + morphing_path if morphing_path else block_path
        
        print(f"Completed search for Component {comp_idx+1}, path length: {len(combined_path)}")
        return (comp_idx, combined_path)

    def search_multi_component(self, time_limit=30):
        """
        Run parallel searches for each component in a multi-component goal
        """
        # Get the initial state
        initial_state = self.start_state
        
        # Create per-component task groups
        component_tasks = []
        
        for comp_idx in range(len(self.goal_components)):
            # Get the blocks assigned to this component
            component_blocks = [pos for pos in self.target_block_list 
                               if self.block_component_assignment.get(pos) == comp_idx]
            
            # Skip empty components
            if not component_blocks:
                print(f"No blocks assigned to Component {comp_idx+1}, skipping")
                continue
                
            # Add a task for this component
            component_tasks.append((comp_idx, component_blocks))
        
        # Set up the pool of workers
        if self.use_multiprocessing and len(component_tasks) > 1:
            print(f"Starting parallel search with {min(self.num_processes, len(component_tasks))} processes")
            
            # Allocate time for each component
            component_time_limit = time_limit / len(component_tasks)
            
            # Create pool and start parallel processing
            with mp.Pool(processes=min(self.num_processes, len(component_tasks))) as pool:
                search_func = partial(self.search_single_component, 
                                      start_state=initial_state, 
                                      time_limit=component_time_limit)
                
                # Start tasks with (comp_idx, blocks) pairs
                results = []
                for comp_idx, blocks in component_tasks:
                    result = pool.apply_async(search_func, args=(comp_idx, blocks))
                    results.append(result)
                
                # Collect all results
                component_paths = []
                for result in results:
                    comp_idx, path = result.get()  # This will wait for the task to complete
                    component_paths.append((comp_idx, path))
                
            # Sort by component index
            component_paths.sort()
            
            # Combine the paths
            combined_path = []
            for comp_idx, path in component_paths:
                if path:
                    # If combined_path is still empty, just add the first path
                    if not combined_path:
                        combined_path = path
                    else:
                        # Otherwise, merge this path with the combined path
                        # We keep the fixed blocks from combined_path and add just the target blocks from this path
                        combined_path = self._merge_component_paths(combined_path, path, comp_idx)
            
            return combined_path
        else:
            # Fall back to sequential search if multiprocessing is disabled
            print("Using sequential search for components")
            return self.search(time_limit)
    
# New method to merge component paths
    def merge_component_paths(self, component_results):
        """Merge paths from different components"""
        if not component_results:
            return []
        
        # Extract valid paths
        valid_paths = []
        for result in component_results:
            if result and isinstance(result, tuple) and len(result) == 2:
                comp_idx, path = result
                if path:  # Only add non-empty paths
                    valid_paths.append((comp_idx, path))
        
        if not valid_paths:
            print("No valid component paths found!")
            return []
        
        # Start with the first component's path
        first_comp_idx, first_path = valid_paths[0]
        merged_path = []
        
        # Copy the first path correctly to start with
        for state in first_path:
            if isinstance(state, frozenset):
                merged_path.append(list(state))
            else:
                merged_path.append(list(state))  # Make a copy of the state list
        
        # Tracking which components have been merged at each step
        merged_components = {i: [first_comp_idx] for i in range(len(merged_path))}
        
        # Merge with each additional component
        for comp_idx, path in valid_paths[1:]:
            print(f"Merging path for Component {comp_idx+1} (length: {len(path)})")
            
            # For each step in the path, update with this component's blocks
            for i in range(min(len(merged_path), len(path))):
                current_state = merged_path[i]
                component_state = path[i]
                
                # Convert to list if needed
                if isinstance(component_state, frozenset):
                    component_state = list(component_state)
                
                # If this component has already been merged for this state, skip
                if comp_idx in merged_components[i]:
                    continue
                
                # Remove blocks from this component in current state
                # (they'll be replaced with the correct positions from component_state)
                filtered_state = []
                for pos in current_state:
                    if (pos not in self.block_component_assignment or 
                        self.block_component_assignment[pos] != comp_idx):
                        filtered_state.append(pos)
                
                # Add blocks from this component's state
                for pos in component_state:
                    if pos in self.block_component_assignment and self.block_component_assignment[pos] == comp_idx:
                        filtered_state.append(pos)
                
                # Update the merged path and track the merged component
                merged_path[i] = filtered_state
                merged_components[i].append(comp_idx)
            
            # If component path is longer, add its remaining states
            if len(path) > len(merged_path):
                for i in range(len(merged_path), len(path)):
                    # Get all blocks except this component from the last merged state
                    base_state = []
                    for pos in merged_path[-1]:
                        if (pos not in self.block_component_assignment or 
                            self.block_component_assignment[pos] != comp_idx):
                            base_state.append(pos)
                    
                    # Add this component's blocks from its current state
                    component_state = path[i]
                    if isinstance(component_state, frozenset):
                        component_state = list(component_state)
                        
                    for pos in component_state:
                        if pos in self.block_component_assignment and self.block_component_assignment[pos] == comp_idx:
                            base_state.append(pos)
                    
                    # Add to the merged path with new component tracking
                    merged_path.append(base_state)
                    # Create new entry in merged_components for this step
                    merged_components[i] = [comp_idx] + merged_components[len(merged_path)-2]
        
        # Final check for missing blocks
        for i, state in enumerate(merged_path):
            # Check if all components are included
            missing_components = []
            for comp_idx in range(len(self.goal_components)):
                if comp_idx not in merged_components[i]:
                    missing_components.append(comp_idx)
            
            # If components are missing, use their positions from the previous state
            if missing_components and i > 0:
                prev_state = merged_path[i-1]
                for pos in prev_state:
                    if (pos in self.block_component_assignment and 
                        self.block_component_assignment[pos] in missing_components and
                        pos not in state):
                        state.append(pos)
                        merged_components[i].append(self.block_component_assignment[pos])
        
        # Final check for overlapping blocks
        for i, state in enumerate(merged_path):
            # Check for duplicates
            if len(state) != len(set(state)):
                print(f"WARNING: Overlapping blocks in merged state {i}, removing duplicates")
                # Fix by removing duplicates
                merged_path[i] = list(set(state))
        
        return merged_path

    def _merge_path_pairs(self, base_path, new_path, comp_idx):
        """
        Merge two paths, keeping fixed blocks from base_path and 
        adding component blocks from new_path
        
        Args:
            base_path: Path containing all blocks except the current component
            new_path: Path containing only the current component's blocks
            comp_idx: Index of the component being merged
            
        Returns:
            Merged path
        """
        if not base_path:
            return new_path
        if not new_path:
            return base_path
            
        # Create a merged path
        merged_path = []
        
        # Use the longer path as the base length
        max_length = max(len(base_path), len(new_path))
        
        for i in range(max_length):
            # Get the current states from both paths (use last state if we've run out)
            base_state = base_path[min(i, len(base_path) - 1)]
            new_state = new_path[min(i, len(new_path) - 1)]
            
            # Create the merged state
            merged_state = []
            
            # Add blocks from base_state that don't belong to this component
            for pos in base_state:
                if pos not in self.block_component_assignment or self.block_component_assignment[pos] != comp_idx:
                    merged_state.append(pos)
            
            # Add blocks from new_state that belong to this component
            for pos in new_state:
                if pos in self.block_component_assignment and self.block_component_assignment[pos] == comp_idx:
                    merged_state.append(pos)
            
            # Check for collisions
            if len(merged_state) != len(set(merged_state)):
                print(f"WARNING: Collision detected in merged state at step {i}")
                # Try to fix by removing duplicates
                merged_state = list(set(merged_state))
            
            merged_path.append(merged_state)
        
        return merged_path

    def smarter_morphing_phase(self, start_state, double time_limit=15):
        """
        Improved Phase 2: Morph the blocks into the goal shape
        With disconnection allowed, only target blocks are morphed
        Fixed blocks remain stationary
        Handles multi-component goals by morphing each component separately
        """
        cdef double start_time
        cdef double best_heuristic, current_heuristic, last_improvement_time, f_score
        cdef int iterations = 0
        cdef int tentative_g
        
        # Skip states with overlapping blocks
        if self.has_overlapping_blocks(start_state):
            print("WARNING: Starting state for morphing has overlapping blocks!")
            return [start_state]
        
        print(f"Starting Smarter Morphing Phase with {self.min_simultaneous_moves}-{self.max_simultaneous_moves} simultaneous moves...")
        if self.allow_disconnection:
            if self.multi_component_goal:
                print(f"Morphing {len(self.target_block_list)} target blocks into {len(self.goal_components)} separate components")
                print(f"Components will be allowed to disconnect from each other")
                for i, component in enumerate(self.goal_components):
                    blocks_for_component = [pos for pos in self.target_block_list 
                                         if pos in self.block_component_assignment and 
                                         self.block_component_assignment[pos] == i]
                    print(f"  Component {i+1}: {len(component)} target positions, assigned {len(blocks_for_component)} blocks")
            else:
                print(f"Morphing only {len(self.target_block_list)} target blocks, keeping {len(self.fixed_block_list)} blocks stationary")
        
        start_time = time.time()
        
        # Initialize beam search
        open_set = [(self.improved_morphing_heuristic(start_state), 0, start_state)]
        closed_set = set()
        
        # Track path, g-scores, and best state
        g_score = {start_state: 0}
        came_from = {start_state: None}
        
        # Track best state seen so far
        best_state = start_state
        best_heuristic = self.improved_morphing_heuristic(start_state)
        last_improvement_time = time.time()
        
        # Determine whether we're targeting a subset of blocks
        targeting_subset = self.allow_disconnection
        
        while open_set and time.time() - start_time < time_limit:
            iterations += 1
            
            # Get state with lowest f-score
            f, g, current = heapq.heappop(open_set)
            
            # Skip if already processed
            if current in closed_set:
                continue
            
            # Skip states with overlapping blocks
            if self.has_overlapping_blocks(current):
                continue
            
            # Check if goal reached
            if targeting_subset:
                # For multi-component goals, we need to check each component individually
                if self.multi_component_goal:
                    # Extract target blocks
                    target_blocks = [pos for pos in current if pos not in self.non_target_state]
                    
                    # Group blocks by their assigned component
                    component_blocks = {}
                    for pos in target_blocks:
                        if pos in self.block_component_assignment:
                            comp_idx = self.block_component_assignment[pos]
                            if comp_idx not in component_blocks:
                                component_blocks[comp_idx] = []
                            component_blocks[comp_idx].append(pos)
                    
                    # Check if each component matches its goal shape or has sufficient matches
                    all_components_match = True
                    total_matches = 0
                    
                    for comp_idx, blocks in component_blocks.items():
                        if comp_idx < len(self.goal_components):
                            goal_component = self.goal_components[comp_idx]
                            
                            # Get goal positions for this component
                            component_set = set(blocks)
                            goal_set = set(goal_component)
                            
                            # Check if this component is in the correct shape
                            if component_set == goal_set:
                                total_matches += len(goal_set)
                                # Print this only occasionally to avoid flooding
                                if iterations % 500 == 0:
                                    print(f"Component {comp_idx+1} perfectly matched")
                            else:
                                # Count partial matches
                                matches = len(component_set.intersection(goal_set))
                                total_matches += matches
                                
                                # If too few matches, this component isn't right
                                if matches < min(len(component_set), len(goal_set)) * 0.7:
                                    all_components_match = False
                    
                    # Goal is reached if all components match exactly or have sufficient matches
                    if all_components_match:
                        print(f"All components matched after {iterations} iterations!")
                        return self.reconstruct_path(came_from, current)
                    
                    # Alternative success: if all goal positions are occupied
                    # Use a lower threshold to increase chance of success
                    if total_matches >= len(self.goal_positions) * 0.90:  # Lowered from 0.95 for better results
                        print(f"Sufficient matches across all components: {total_matches}/{len(self.goal_positions)}")
                        return self.reconstruct_path(came_from, current)
                else:
                    # For single-component goals with disconnection allowed
                    target_blocks = frozenset(pos for pos in current if pos not in self.non_target_state)
                    
                    # Check if all goal positions are filled by target blocks
                    if target_blocks == self.goal_state:
                        print(f"Goal state reached after {iterations} iterations!")
                        return self.reconstruct_path(came_from, current)
                    
                    # Alternative goal check: if enough blocks are in the right positions
                    matching_positions = len(target_blocks.intersection(self.goal_state))
                    if matching_positions == len(self.goal_state):
                        print(f"Goal positions matched after {iterations} iterations!")
                        return self.reconstruct_path(came_from, current)
            else:
                # For exact matching, use original goal check
                if current == self.goal_state:
                    print(f"Goal reached after {iterations} iterations!")
                    return self.reconstruct_path(came_from, current)
            
            # Check if this is the best state seen so far
            current_heuristic = self.improved_morphing_heuristic(current)
            if current_heuristic < best_heuristic:
                best_state = current
                best_heuristic = current_heuristic
                last_improvement_time = time.time()
                
                # Print progress occasionally
                if iterations % 500 == 0:
                    print(f"Progress: h={best_heuristic}, iterations={iterations}")
            
            # Check for stagnation
            if time.time() - last_improvement_time > time_limit * 0.3:
                print("Search stagnated, restarting from best state...")
                # Clear the beam and start from the best state
                open_set = [(best_heuristic, g_score[best_state], best_state)]
                last_improvement_time = time.time()
            
            # Limit iterations to prevent infinite loops
            if iterations >= self.max_iterations:
                print(f"Reached max iterations ({self.max_iterations})")
                break
            
            closed_set.add(current)
            
            # Get all valid moves
            neighbors = self.get_all_valid_moves(current)
            
            # Process each neighbor
            for neighbor in neighbors:
                # Skip states with overlapping blocks
                if self.has_overlapping_blocks(neighbor):
                    continue
                
                # Additional validation for states with fewer blocks in goal
                if self.allow_disconnection:
                    # Extract only target blocks
                    target_blocks = [pos for pos in neighbor if pos not in self.non_target_state]
                    
                    # Skip if we don't have the right number of target blocks
                    if len(target_blocks) != len(self.goal_positions):
                        continue
                    
                    # Skip if target blocks overlap with each other
                    if len(target_blocks) != len(set(target_blocks)):
                        continue
                    
                    # Skip if any target block occupies the same position as a fixed block
                    fixed_blocks = [pos for pos in neighbor if pos in self.non_target_state]
                    if any(pos in fixed_blocks for pos in target_blocks):
                        continue
                    
                    # For multi-component goals with disconnection allowed:
                    # Only check connectivity WITHIN each component, not BETWEEN components
                    if self.multi_component_goal:
                        # Important: We've modified check_component_connectivity to allow disconnections between components
                        if not self.check_component_connectivity(neighbor):
                            continue
                    # For single-component goals with disconnection allowed:
                    # Ensure target blocks stay connected to each other
                    elif not (self.is_connected(target_blocks) or len(target_blocks) <= 1):
                        continue
                
                if neighbor in closed_set:
                    continue
                
                # Calculate tentative g-score
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # This is a better path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.improved_morphing_heuristic(neighbor)
                    
                    # Add to open set
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
            
            # Beam search pruning: keep only the best states
            if len(open_set) > self.beam_width:
                open_set = heapq.nsmallest(self.beam_width, open_set)
                heapq.heapify(open_set)
        
        # If we exit the loop, either no path was found or time limit reached
        if time.time() - start_time >= time_limit:
            print(f"Morphing phase timed out after {iterations} iterations!")
        
        # Print the best state's heuristic value
        best_h = self.improved_morphing_heuristic(best_state)
        print(f"Best state found with heuristic value: {best_h}")
        
        # For multi-component goals, check component matching
        if self.allow_disconnection and self.multi_component_goal:
            # Extract target blocks
            target_blocks = [pos for pos in best_state if pos not in self.non_target_state]
            
            # Group blocks by their assigned component
            component_blocks = {}
            for pos in target_blocks:
                if pos in self.block_component_assignment:
                    comp_idx = self.block_component_assignment[pos]
                    if comp_idx not in component_blocks:
                        component_blocks[comp_idx] = []
                    component_blocks[comp_idx].append(pos)
            
            # Print statistics for each component
            for comp_idx, blocks in component_blocks.items():
                if comp_idx < len(self.goal_components):
                    goal_component = self.goal_components[comp_idx]
                    matching_positions = sum(1 for pos in blocks if pos in goal_component)
                    print(f"Component {comp_idx+1}: {matching_positions}/{len(goal_component)} matching positions")
        
        # Return the best state found
        return self.reconstruct_path(came_from, best_state)

    def reconstruct_path(self, came_from, current):
        """
        Reconstruct the path from start to goal
        """
        path = []
        while current:
            # Convert frozenset to list
            if isinstance(current, frozenset):
                path.append(list(current))
            else:
                path.append(list(current))
                
            current = came_from.get(current)
        
        path.reverse()
        return path
    def _generate_component_separation_moves(self, state):
        """
        Generate special moves that encourage components to separate
        This is critical for allowing blocks to disconnect into separate shapes
        """
        valid_moves = []
        
        # Skip if not a multi-component goal with disconnection allowed
        if not (self.multi_component_goal and self.allow_disconnection):
            return valid_moves
        
        # Convert state to list if needed
        state_list = list(state) if isinstance(state, frozenset) else state
        
        # Group blocks by component
        component_blocks = {}
        for pos in state_list:
            if pos in self.block_component_assignment:
                comp_idx = self.block_component_assignment[pos]
                if comp_idx not in component_blocks:
                    component_blocks[comp_idx] = []
                component_blocks[comp_idx].append(pos)
        
        # For each component, try moving blocks toward their goal centroid
        for comp_idx, blocks in component_blocks.items():
            if len(blocks) <= 0 or comp_idx >= len(self.goal_component_centroids):
                continue
                
            goal_centroid = self.goal_component_centroids[comp_idx]
            
            # Find the block in this component that's farthest from the goal
            farthest_block = None
            max_dist = -1
            
            for pos in blocks:
                dist = abs(pos[0] - goal_centroid[0]) + abs(pos[1] - goal_centroid[1])
                if dist > max_dist:
                    max_dist = dist
                    farthest_block = pos
            
            if not farthest_block:
                continue
                
            # Try to move this block closer to the goal centroid
            dx = 1 if goal_centroid[0] > farthest_block[0] else -1 if goal_centroid[0] < farthest_block[0] else 0
            dy = 1 if goal_centroid[1] > farthest_block[1] else -1 if goal_centroid[1] < farthest_block[1] else 0
            
            # Try the move
            new_pos = (farthest_block[0] + dx, farthest_block[1] + dy)
            
            # Skip if out of bounds
            if not (0 <= new_pos[0] < self.grid_size[0] and 0 <= new_pos[1] < self.grid_size[1]):
                continue
                
            # Skip if already occupied
            if new_pos in state_list:
                continue
                
            # Create a new state with this block moved
            new_state = [pos if pos != farthest_block else new_pos for pos in state_list]
            
            # Check if the component remains connected after the move
            comp_blocks_after = [pos for pos in new_state 
                               if pos in self.block_component_assignment and 
                               self.block_component_assignment[pos] == comp_idx]
            
            # CRITICAL: Use is_connected only for this component's blocks
            if len(comp_blocks_after) <= 1 or self.is_connected(comp_blocks_after):
                valid_moves.append(frozenset(new_state))
            
            # If that failed, try moving the component as a whole
            if not valid_moves:
                # Try moving all blocks in this component together
                new_positions = [(pos[0] + dx, pos[1] + dy) for pos in blocks]
                
                # Check if all new positions are valid
                all_valid = True
                other_blocks = [pos for pos in state_list if pos not in blocks]
                
                for pos in new_positions:
                    # Check bounds
                    if not (0 <= pos[0] < self.grid_size[0] and 0 <= pos[1] < self.grid_size[1]):
                        all_valid = False
                        break
                        
                    # Check collision with other blocks
                    if pos in other_blocks:
                        all_valid = False
                        break
                
                if all_valid:
                    # Create a new state with these blocks moved
                    new_state = other_blocks + new_positions
                    valid_moves.append(frozenset(new_state))
        
        return valid_moves

# Override search method
    def search(self, double time_limit=30):
        """Main search method with special handling for disconnected goals"""
        # For disconnected goals, use component-specific search
        if self.multi_component_goal and self.allow_disconnection:
            print("Using component-specific search for disconnected goal shapes")
            
            # CRITICAL FIX: Force disconnection to be allowed
            self.allow_disconnection = True
            
            # First, validate component assignments
            self._validate_component_assignments()
            
            # Run the specialized multi-component search
            component_path = self.search_parallel_components(time_limit)
            
            # Ensure the path has no overlapping blocks
            component_path = self._fix_overlapping_blocks_in_path(component_path)
            
            return component_path
        
        # Otherwise use the standard search approach
        # Allocate time for each phase
        cdef double block_time_limit = time_limit * 0.3  # 30% for block movement
        cdef double morphing_time_limit = time_limit * 0.7  # 70% for morphing
        
        # Phase 1: Block Movement
        block_path = self.block_movement_phase(block_time_limit)
        
        if not block_path:
            print("Block movement phase failed!")
            return None
        
        # Verify no overlapping blocks
        block_final_list = block_path[-1]
        if len(block_final_list) != len(set(block_final_list)):
            print("WARNING: Block movement produced a state with overlapping blocks!")
            # Try to fix it by removing duplicates
            block_final_list = list(set(block_final_list))
            block_path[-1] = block_final_list
        
        # Get the final state from block movement phase
        block_final_state = frozenset(block_final_list)
        
        # Phase 2: Smarter Morphing
        morphing_path = self.smarter_morphing_phase(block_final_state, morphing_time_limit)
        
        if not morphing_path:
            print("Morphing phase failed!")
            return block_path
        
        # Combine paths (remove duplicate state at transition)
        combined_path = block_path[:-1] + morphing_path
        
        # Final check for overlapping blocks
        for i, state in enumerate(combined_path):
            if len(state) != len(set(state)):
                print(f"WARNING: State {i} in path has overlapping blocks!")
        
        return combined_path
               
    def _fix_overlapping_blocks_in_path(self, path):
        """
        Fix any overlapping blocks in the path by removing duplicates
        This is a last resort to ensure no overlapping blocks in the final path
        """
        fixed_path = []
        for state in path:
            # Convert to set to remove duplicates
            fixed_state = list(set(state))
            
            # If we lost blocks in the conversion, print a warning
            if len(fixed_state) != len(state):
                print(f"WARNING: Removed {len(state) - len(fixed_state)} duplicate blocks in state")
                
            fixed_path.append(fixed_state)
            
        return fixed_path 