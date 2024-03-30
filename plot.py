import matplotlib.pyplot as plt


def plot_points(points_list, label_names=None):
    """
    Plot points from a list of lists of points on the same plot, using indices as x-values.

    Parameters:
        points_list (list): A list of lists of points. Each inner list contains numbers representing the points.
        label_names (list): A list of label names corresponding to each set of points.

    Returns:
        None
    """
    # Create a new figure
    plt.figure()

    # Iterate through each list of points
    for i, points in enumerate(points_list):
        # Plot the points
        plt.plot(range(len(points)), points, marker='o', label=label_names[i] if label_names else f'Set {i + 1}')

    # Set plot labels and title
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Model Loss over Training')

    # Add legend
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()


# # Example usage:
# points_list = [
#     [1, 2, 3, 4],
#     [2, 4, 6, 8],
#     [3, 6, 9, 12]
# ]
# label_names = ['A', 'B', 'C']
#
# plot_points(points_list, label_names)
