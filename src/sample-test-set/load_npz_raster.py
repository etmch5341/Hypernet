import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

def display_raster_with_goals(raster, goal_points, title="Raster with Goal Points", output_file=None):
    """
    Display the raster with goal points highlighted.
    
    Parameters:
    - raster: numpy array representing the cost map
    - goal_points: list of tuples (x, y, name) in pixel coordinates
    - title: plot title
    - output_file: path to save the plot (if None, shows interactively)
    """
    plt.figure(figsize=(12, 10))
    
    # Display the raster
    plt.imshow(raster, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Cost Value')
    
    # Overlay goal points with red markers
    for x, y, name in goal_points:
        plt.plot(x, y, 'r*', markersize=20, markeredgecolor='white', markeredgewidth=2)
        plt.annotate(name, (x, y), xytext=(10, 10), textcoords='offset points',
                    color='white', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.7))
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()
    
    plt.close()

# Save plots to the outputs directory (mounted volume)
data = np.load('./src/sample-test-set/austin_test_raster.npz', allow_pickle=True)
loaded_raster = data['raster']
loaded_goal_points = data['goal_points']
display_raster_with_goals(loaded_raster, loaded_goal_points, 
                          title="Austin Test Raster with Goal Points",
                          output_file='/workspace/outputs/austin_test_raster.png')

data = np.load('./src/sample-test-set/seattle_test_raster.npz', allow_pickle=True)
loaded_raster = data['raster']
loaded_goal_points = data['goal_points']
display_raster_with_goals(loaded_raster, loaded_goal_points, 
                          title="Seattle Test Raster with Goal Points",
                          output_file='/workspace/outputs/seattle_test_raster.png')

data = np.load('./src/sample-test-set/portland_test_raster.npz', allow_pickle=True)
loaded_raster = data['raster']
loaded_goal_points = data['goal_points']
display_raster_with_goals(loaded_raster, loaded_goal_points, 
                          title="Portland Test Raster with Goal Points",
                          output_file='/workspace/outputs/portland_test_raster.png')