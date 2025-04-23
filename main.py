import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better stability
import matplotlib.pyplot as plt
import sys
from Controller import SearchController

if __name__ == "__main__":
    try:
        grid_size = (10, 10)
        
        # Dictionary of formations
        formations = {
            "start": [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3), (0, 4), (1, 4),
                      (0, 5), (1, 5), (0, 6), (1, 6), (0, 7), (1, 7), (0, 8), (1, 8), (0, 9), (1, 9)],
            
            "Ring": [(7, 4), (7, 5), (6, 3), (6, 4), (6, 5), (6, 6), (5, 2), (5, 3), (5, 6), (5, 7),
                     (4, 2), (4, 3), (4, 6), (4, 7), (3, 3), (3, 4), (3, 5), (3, 6), (2, 4), (2, 5)],
            
            "Rectangle": [(3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7),
                          (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7)],
            
            "Triangle": [(2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (3, 2), (3, 3),
                         (3, 4), (3, 5), (3, 6), (3, 7), (4, 3), (4, 4), (4, 5), (4, 6), (5, 4), (5, 5)]
        }
        
        # Create controller with the right parameters
        controller = SearchController(
            grid_size=grid_size, 
            formations=formations, 
            topology="moore", 
            time_limit=1000,
            max_simultaneous_moves=1,
            min_simultaneous_moves=1
        )

        # Disable interactive mode and show the window
        plt.ioff()
        plt.show()

    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(traceback.format_exc())
        sys.exit(1)