


import torch
import matplotlib.pyplot as plt
import numpy as np


def compute_non_overlapping_ray_radii(ray_directions, ray_indices=None, base_radius=1.1,
                                      min_angular_separation=15.0, radius_increment=0.1):
    """
    Compute radii for ray labels to avoid overlaps when rays are close together.

    Parameters:
    -----------
    ray_directions : torch.Tensor or array-like
        n x 2 tensor/array of ray direction vectors
    ray_indices : list, optional
        Indices of rays to display (1-based). If None, uses all rays.
    base_radius : float
        Starting radius for ray labels. Default 1.1.
    min_angular_separation : float
        Minimum angular separation in degrees below which labels need spacing. Default 15.0.
    radius_increment : float
        Amount to increment radius for overlapping labels. Default 0.1.

    Returns:
    --------
    dict : Mapping from ray numbers to radii
    """
    # Convert to numpy
    if isinstance(ray_directions, torch.Tensor):
        ray_dirs = ray_directions.numpy()
    else:
        ray_dirs = np.array(ray_directions)

    # Filter rays if needed
    if ray_indices is not None:
        selected_indices = [i - 1 for i in ray_indices]
        ray_dirs = ray_dirs[selected_indices]
        ray_numbers = list(ray_indices)
    else:
        ray_numbers = list(range(1, len(ray_dirs) + 1))

    # Normalize directions
    norms = np.linalg.norm(ray_dirs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    ray_dirs_normalized = ray_dirs / norms

    # Compute angles for each ray
    angles = np.arctan2(ray_dirs_normalized[:, 1], ray_dirs_normalized[:, 0])
    angles_deg = np.degrees(angles)

    # Sort rays by angle
    sorted_indices = np.argsort(angles_deg)

    # Assign radii to avoid overlaps
    radii = {}
    current_radius = base_radius

    for i, idx in enumerate(sorted_indices):
        ray_num = ray_numbers[idx]

        if i == 0:
            # First ray gets base radius
            radii[ray_num] = base_radius
            current_radius = base_radius
        else:
            # Check angular separation from previous ray
            prev_idx = sorted_indices[i - 1]
            angular_diff = abs(angles_deg[idx] - angles_deg[prev_idx])

            # Handle wrap-around at 360 degrees
            if angular_diff > 180:
                angular_diff = 360 - angular_diff

            if angular_diff < min_angular_separation:
                # Too close - increment radius
                current_radius += radius_increment
                radii[ray_num] = current_radius
            else:
                # Far enough - reset to base radius
                current_radius = base_radius
                radii[ray_num] = base_radius

    return radii




def get_label_colors(label_list, colormap='hsv'):
    """
    Generate colors for labels using the same formula as plot_unit_circle_scatter_labeled.

    Parameters:
    -----------
    label_list : list
        List of label names
    colormap : str
        Matplotlib colormap name (default: 'hsv')

    Returns:
    --------
    colors : list
        List of RGBA color tuples, one per label
    colors_dict : dict
        Dictionary mapping label names to RGBA colors

    Example:
    --------
    >>> label_list = ['A - Std Normal', 'B - Correlated Normal', 'C - Uniform']
    >>> colors, colors_dict = get_label_colors(label_list, colormap='hsv')
    >>> # colors[0] is the color for 'A - Std Normal'
    >>> # colors_dict['B - Correlated Normal'] gives the color for that label
    """
    cmap = plt.get_cmap(colormap)
    num_labels = len(label_list)

    # Use the same formula as plot_unit_circle_scatter_labeled (line 625)
    colors = [cmap(i / num_labels) for i in range(num_labels)]

    # Also create a dictionary for easy lookup
    colors_dict = {label: colors[i] for i, label in enumerate(label_list)}

    return colors, colors_dict



def plot_tensor_bars(tensor_data, figsize=(10, 6), title='Bar Chart from Tensor',
                     xlabel='Index', ylabel='Value', show_grid=True, separate_plots=True, labels=None,
                     start_index=0, cmap='tab10', cmap_based_on='index', num_colors=None, row_colors=None):
    """
    Create a bar chart from a 1D or 2D PyTorch tensor.
    
    Parameters:
    -----------
    tensor_data : torch.Tensor
        1D tensor for single bar chart, or 2D tensor for multiple subplots (one per row)
    figsize : tuple
        Figure size (width, height)
    title : str
        Chart title (for 1D) or base title (for 2D, will append row number)
    xlabel : str
        Label for x-axis
    ylabel : str
        Label for y-axis
    show_grid : bool
        Whether to show grid lines
    separate_plots : bool
        For 2D tensors: if True, create separate subplots; if False, overlay on same chart
    labels : list of str, optional
        Labels for each row in 2D tensor. If None, uses 'Row 0', 'Row 1', etc.
    start_index : int, optional
        Starting index for x-axis labels. Default 0. Use 1 to start numbering from 1.
    cmap : str or Colormap, optional
        Colormap name (e.g., 'viridis', 'plasma', 'coolwarm', 'tab10') or Colormap object.
        If None, uses default colors.
        For 2D overlaid plots (separate_plots=False), colors are assigned to match
        plot_unit_circle_scatter_labeled behavior.
    cmap_based_on : str, optional
        What to base colormap on: 'value' (bar heights) or 'index' (bar positions).
        Default is 'value'. Only applies to 1D and 2D separate plots when cmap is specified.
    num_colors : int, optional
        Number of colors to use for dividing the colormap. This ensures consistent color
        assignment across plots. If None, uses num_rows for 2D overlaid plots.
        Example: If you have 3 categories in a scatter plot, use num_colors=3 in the bar plot
        to ensure the first 3 bars match the scatter plot colors, even if bar plot has more rows.
    row_colors : list of colors, optional
        Explicit colors to use for each row (for 2D overlaid plots). Can be a list of:
        - Color names (e.g., ['red', 'blue', 'green'])
        - RGB tuples (e.g., [(1.0, 0, 0), (0, 0, 1.0)])
        - RGBA tuples (e.g., [(1.0, 0, 0, 0.8), (0, 0, 1.0, 0.8)])
        If provided, overrides cmap and num_colors for color selection.
        Length must match number of rows in tensor_data.
     """
    # Handle both 1D and 2D tensors
    if tensor_data.dim() == 1:
        # Single row - create one bar chart
        values = tensor_data.numpy()
        x_positions = range(len(values))
        x_labels = range(start_index, start_index + len(values))

        plt.figure(figsize=figsize)
        plt.bar(x_positions, values)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(x_positions, x_labels)
        if show_grid:
            plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    elif tensor_data.dim() == 2:
        # Multiple rows
        num_rows = len(tensor_data)
        
        # Generate default labels if not provided
        if labels is None:
            labels = [f'Row {i}' for i in range(num_rows)]
        elif len(labels) != num_rows:
            raise ValueError(f"Number of labels ({len(labels)}) must match number of rows ({num_rows})")
        
        if separate_plots:
            # Create separate subplots for each row
            fig, axes = plt.subplots(num_rows, 1, figsize=(figsize[0], figsize[1]*num_rows/2))

            # Handle case where there's only one row (axes won't be an array)
            if num_rows == 1:
                axes = [axes]

            for i, row in enumerate(tensor_data):
                values = row.numpy()
                x_positions = range(len(values))
                x_labels = range(start_index, start_index + len(values))

                # Apply colormap if specified
                if cmap is not None:
                    if cmap_based_on == 'value':
                        norm = Normalize(vmin=values.min(), vmax=values.max())
                        colors = cm.get_cmap(cmap)(norm(values))
                    else:  # 'index'
                        norm = Normalize(vmin=0, vmax=len(values)-1)
                        colors = cm.get_cmap(cmap)(norm(x_positions))
                    axes[i].bar(x_positions, values, color=colors)
                else:
                    axes[i].bar(x_positions, values)

                axes[i].bar(x_positions, values)
                axes[i].set_title(f'{title} - {labels[i]}')
                axes[i].set_xlabel(xlabel)
                axes[i].set_ylabel(ylabel)
                axes[i].set_xticks(x_positions, x_labels)
                if show_grid:
                    axes[i].grid(axis='y', alpha=0.3)
        else:
            # Plot all rows on the same chart
            plt.figure(figsize=figsize)
            width = 0.8 / num_rows  # Divide bar width among rows
            n_values = len(tensor_data[0])
            x_positions = range(n_values)
            x_labels = range(start_index, start_index + n_values)

            # Determine colors for each row
            if row_colors is not None:
                # Use explicitly provided colors
                if len(row_colors) != num_rows:
                    raise ValueError(f"Length of row_colors ({len(row_colors)}) must match number of rows ({num_rows})")
                colors_to_use = row_colors
            elif cmap is not None:
                # Generate colors from colormap
                colormap_obj = plt.get_cmap(cmap)
                # Use num_colors if specified, otherwise use num_rows
                color_divisor = num_colors if num_colors is not None else num_rows
                # Sample colormap at positions matching the scatter plot formula
                colors_to_use = [colormap_obj(i / color_divisor) for i in range(num_rows)]
            else:
                # Use default matplotlib colors
                colors_to_use = [None] * num_rows

            for i, row in enumerate(tensor_data):
                values = row.numpy()
                x_pos = [x + (i - num_rows/2 + 0.5) * width for x in x_positions]
                bar_color = colors_to_use[i]
                plt.bar(x_pos, values, width=width, label=labels[i], alpha=0.8, color=bar_color)

            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.xticks(x_positions, x_labels)
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            if show_grid:
                plt.grid(axis='y', alpha=0.3)
                
        plt.tight_layout()
        plt.show()
    else:
        raise ValueError("Input tensor must be 1D or 2D")

_ = '''
# Example usage
if __name__ == "__main__":
    # 1D tensor example
    tensor_1d = torch.tensor([0.0981, 0.0981, 0.0981, 0.1045, 0.0981, 0.0981, 
                              0.1045, 0.1045, 0.0981, 0.0981], dtype=torch.float64)
    plot_tensor_bars(tensor_1d)
    
    # 2D tensor example - separate plots
    tensor_2d = torch.tensor([[0.0981, 0.0981, 0.0981, 0.1045],
                              [0.0981, 0.0981, 0.1045, 0.1045],
                              [0.1045, 0.0981, 0.0981, 0.0981]])
    plot_tensor_bars(tensor_2d, title='Multi-row Tensor')
    
    # 2D tensor example - same chart with custom labels
    custom_labels = ['Model A', 'Model B', 'Model C']
    plot_tensor_bars(tensor_2d, title='Multi-row Tensor (Overlaid)',
                    separate_plots=False, labels=custom_labels)
    '''


def plot_unit_circle_scatter(tensor_data, figsize=(8, 8), title='Unit Circle Scatter Plot',
                             point_size=20, point_alpha=0.6, point_color='blue',
                             show_circle=True, show_grid=True, show_axes=True):
    """
    Create a scatter plot on the unit circle by normalizing all vectors in an n x 2 tensor.

    Parameters:
    -----------
    tensor_data : torch.Tensor
        n x 2 tensor where each row is a 2D vector to be normalized and plotted
    figsize : tuple
        Figure size (width, height)
    title : str
        Plot title
    point_size : float
        Size of scatter points
    point_alpha : float
        Transparency of points (0-1)
    point_color : str or array-like
        Color(s) for the scatter points
    show_circle : bool
        Whether to draw the unit circle
    show_grid : bool
        Whether to show grid lines
    show_axes : bool
        Whether to show x and y axes through origin

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    if tensor_data.dim() != 2 or tensor_data.shape[1] != 2:
        raise ValueError("Input tensor must be n x 2 (each row is a 2D vector)")

    # Convert to numpy for processing
    vectors = tensor_data.numpy()

    # Normalize each vector to unit length
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Handle zero vectors by keeping them at origin
    norms = np.where(norms == 0, 1, norms)
    normalized = vectors / norms

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the unit circle
    if show_circle:
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)
        ax.plot(circle_x, circle_y, 'k-', linewidth=1.5, alpha=0.3, label='Unit Circle')

    # Scatter plot of normalized vectors
    ax.scatter(normalized[:, 0], normalized[:, 1],
              s=point_size, alpha=point_alpha, c=point_color, edgecolors='black', linewidth=0.5)

    # Show axes through origin
    if show_axes:
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)

    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

    # Labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)

    # Grid
    if show_grid:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig, ax


def plot_unit_circle_scatter_labeled(tensor_data, labels=None, index_tensor=None,
                                     label_list=None, label_list_row_index_lookup_dict=None,
                                     label_radii=None,
                                     label_colors=None,
                                     figsize=(8, 8),
                                     title='Unit Circle Scatter Plot (Labeled)',
                                     point_size=20, point_alpha=0.6, colormap='tab10',
                                     show_circle=True, show_grid=True, show_axes=True,
                                     show_legend=True, circle_colors=None, reference_circles=None,
                                     ray_directions=None, ray_label_radius=None, ray_indices=None,
                                     ray_color='gray', ray_linewidth=1.0, ray_alpha=0.5,
                                     ray_label_fontsize=10):
    """
    Create a scatter plot on the unit circle with labeled data points.
    Different labels are shown in different colors and can be plotted at different radii.

    Parameters:
    -----------
    tensor_data : torch.Tensor
        n x 2 tensor where each row is a 2D vector to be normalized and plotted
    labels : torch.Tensor or array-like, optional
        1D tensor/array of length n with integer or string labels for each point.
    index_tensor : torch.Tensor, optional
        1D tensor of label indices. Used with label_list to map indices to label names.
    label_list : list, optional
        List of unique label names/values. Used with index_tensor or label_list_row_index_lookup_dict.
    label_list_row_index_lookup_dict : dict, optional
        Dictionary mapping each label to list of row indices in tensor_data.
        Keys are labels from label_list, values are lists of integer indices.
    label_radii : dict or list or float, optional
        Radius for each label. Can be:
        - dict mapping label names to radius values (e.g., {'A': 1.0, 'B': 0.8})
        - list of radii (same order as label_list)
        - single float to use for all labels
        Default is 1.0 for all labels (unit circle)
    label_colors : list or dict of colors, optional                                                   │ │
        Explicit colors to use for each label. Can be:                                                │ │
        - List of colors (same order as unique_labels)                                                │ │
        - Dict mapping label names to colors (e.g., {'A': 'red', 'B': 'blue'})                        │ │
        Colors can be names, RGB tuples, or RGBA tuples.                                              │ │
        If provided, overrides colormap for color selection.
    figsize : tuple
        Figure size (width, height)
    title : str
        Plot title
    point_size : float
        Size of scatter points
    point_alpha : float
        Transparency of points (0-1)
    colormap : str
        Matplotlib colormap name for coloring different labels
    show_circle : bool
        Whether to draw circles at the radii used by labels. Default True.
    show_grid : bool
        Whether to show grid lines
    show_axes : bool
        Whether to show x and y axes through origin
    show_legend : bool
        Whether to show legend for labels
    circle_colors : dict or str, optional
        Colors for circles at label radii. Can be:
        - dict mapping radius values to color strings (e.g., {1.0: 'blue', 0.5: 'red'})
        - single color string to use for all circles
        Default is 'black' with alpha=0.3
    reference_circles : list or dict, optional
        Additional reference circles to draw. Can be:
        - list of radius values (e.g., [0.5, 1.0, 1.5])
        - dict mapping radius values to colors (e.g., {0.5: 'blue', 1.5: 'red'})
        These are drawn regardless of show_circle setting.
    ray_directions : torch.Tensor or array-like, optional
        n x 2 tensor/array defining directions for numbered rays from origin.
        Each row defines a direction vector. Rays are numbered starting from 1
        corresponding to row index + 1.
    ray_label_radius : float, dict, or list, optional
        Distance from origin to place ray number labels. Can be:
        - float: same radius for all rays (if None, uses max_radius * 1.1)
        - dict: mapping ray numbers to radii (e.g., {1: 1.0, 2: 1.2, 3: 1.1})
        - list: radii for each ray in order (length must match number of displayed rays)
    ray_indices : list or array-like, optional
        List of ray indices (1-based) to display. If None, shows all rays.
        E.g., [1, 3, 5] shows only rays 1, 3, and 5 from ray_directions.
    ray_color : str, optional
        Color for the ray lines. Default 'gray'.
    ray_linewidth : float, optional
        Line width for rays. Default 1.0.
    ray_alpha : float, optional
        Transparency for rays. Default 0.5.
    ray_label_fontsize : int, optional
        Font size for ray number labels. Default 10.

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    if tensor_data.dim() != 2 or tensor_data.shape[1] != 2:
        raise ValueError("Input tensor must be n x 2 (each row is a 2D vector)")

    # Convert to numpy for processing
    vectors = tensor_data.numpy()

    # Handle different label input formats
    if labels is not None:
        # Direct labels array provided
        if isinstance(labels, torch.Tensor):
            labels_array = labels.numpy()
        else:
            labels_array = np.array(labels)

        if len(labels_array) != len(vectors):
            raise ValueError(f"Number of labels ({len(labels_array)}) must match number of vectors ({len(vectors)})")

        # Get unique labels for plotting
        unique_labels = np.unique(labels_array)

    elif index_tensor is not None and label_list is not None:
        # index_tensor and label_list provided (output from data_df_to_pytorch_data_tensors_and_labels)
        if isinstance(index_tensor, torch.Tensor):
            indices = index_tensor.numpy()
        else:
            indices = np.array(index_tensor)

        if len(indices) != len(vectors):
            raise ValueError(f"Number of indices ({len(indices)}) must match number of vectors ({len(vectors)})")

        # Map indices to label names
        labels_array = np.array([label_list[i] for i in indices])
        unique_labels = label_list

    elif label_list is not None and label_list_row_index_lookup_dict is not None:
        # Label list and lookup dict provided - reconstruct labels array
        labels_array = np.empty(len(vectors), dtype=object)
        unique_labels = label_list

        for label in label_list:
            if label in label_list_row_index_lookup_dict:
                indices = label_list_row_index_lookup_dict[label]
                labels_array[indices] = label

    else:
        raise ValueError("Must provide either 'labels', 'index_tensor' with 'label_list', or both 'label_list' and 'label_list_row_index_lookup_dict'")

    # Process label_radii parameter
    radii_dict = {}
    if label_radii is None:
        # Default: all labels at radius 1.0
        for label in unique_labels:
            radii_dict[label] = 1.0
    elif isinstance(label_radii, dict):
        # Dictionary mapping labels to radii
        radii_dict = label_radii
        # Fill in missing labels with 1.0
        for label in unique_labels:
            if label not in radii_dict:
                radii_dict[label] = 1.0
    elif isinstance(label_radii, (list, tuple, np.ndarray)):
        # List of radii (same order as unique_labels)
        if len(label_radii) != len(unique_labels):
            raise ValueError(f"Length of label_radii ({len(label_radii)}) must match number of unique labels ({len(unique_labels)})")
        for i, label in enumerate(unique_labels):
            radii_dict[label] = label_radii[i]
    elif isinstance(label_radii, (int, float)):
        # Single radius for all labels
        for label in unique_labels:
            radii_dict[label] = float(label_radii)
    else:
        raise ValueError(f"label_radii must be dict, list, or numeric, got {type(label_radii)}")

    # Normalize each vector to unit length, then scale by label-specific radius
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Handle zero vectors by keeping them at origin
    norms = np.where(norms == 0, 1, norms)
    normalized = vectors / norms

    # Scale each point by its label's radius
    scaled_vectors = np.empty_like(normalized)
    for i, label in enumerate(labels_array):
        scaled_vectors[i] = normalized[i] * radii_dict[label]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Plot circles for each unique radius used by labels
    if show_circle:
        unique_radii = sorted(set(radii_dict.values()))

        # Process circle_colors parameter
        if circle_colors is None:
            # Default color for all circles
            colors_dict = {r: 'k' for r in unique_radii}
        elif isinstance(circle_colors, str):
            # Single color for all circles
            colors_dict = {r: circle_colors for r in unique_radii}
        elif isinstance(circle_colors, dict):
            # Dictionary mapping radii to colors
            colors_dict = {}
            for radius in unique_radii:
                colors_dict[radius] = circle_colors.get(radius, 'k')
        else:
            raise ValueError(f"circle_colors must be dict or str, got {type(circle_colors)}")

        for radius in unique_radii:
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = radius * np.cos(theta)
            circle_y = radius * np.sin(theta)
            ax.plot(circle_x, circle_y, color=colors_dict[radius], linewidth=1.5, alpha=0.3)

    # Plot reference circles
    if reference_circles is not None:
        if isinstance(reference_circles, (list, tuple)):
            # List of radii - use default color
            for radius in reference_circles:
                theta = np.linspace(0, 2*np.pi, 100)
                circle_x = radius * np.cos(theta)
                circle_y = radius * np.sin(theta)
                ax.plot(circle_x, circle_y, 'k--', linewidth=1.5, alpha=0.3)
        elif isinstance(reference_circles, dict):
            # Dictionary mapping radii to colors
            for radius, color in reference_circles.items():
                theta = np.linspace(0, 2*np.pi, 100)
                circle_x = radius * np.cos(theta)
                circle_y = radius * np.sin(theta)
                ax.plot(circle_x, circle_y, color=color, linestyle='--', linewidth=1.5, alpha=0.3)
        else:
            raise ValueError(f"reference_circles must be list or dict, got {type(reference_circles)}")

    # Process and draw rays if provided
    if ray_directions is not None:
        # Convert to numpy if needed
        if isinstance(ray_directions, torch.Tensor):
            ray_dirs = ray_directions.numpy()
        else:
            ray_dirs = np.array(ray_directions)

        if ray_dirs.ndim != 2 or ray_dirs.shape[1] != 2:
            raise ValueError("ray_directions must be n x 2 (each row is a 2D direction vector)")

        # Filter rays if ray_indices is provided
        if ray_indices is not None:
            if isinstance(ray_indices, (list, tuple, np.ndarray)):
                selected_indices = [i - 1 for i in ray_indices]  # Convert 1-based to 0-based
                ray_dirs = ray_dirs[selected_indices]
                ray_numbers = list(ray_indices)  # Use the original 1-based numbers
            else:
                raise ValueError("ray_indices must be a list, tuple, or array")
        else:
            ray_numbers = list(range(1, len(ray_dirs) + 1))  # Default: all rays numbered from 1

        # Normalize ray directions
        ray_norms = np.linalg.norm(ray_dirs, axis=1, keepdims=True)
        ray_norms = np.where(ray_norms == 0, 1, ray_norms)
        ray_dirs_normalized = ray_dirs / ray_norms

        # Determine max radius for determining ray extent
        max_radius_for_rays = max(radii_dict.values()) if radii_dict else 1.0
        if reference_circles is not None:
            if isinstance(reference_circles, (list, tuple)):
                ref_max = max(reference_circles) if reference_circles else 0
            elif isinstance(reference_circles, dict):
                ref_max = max(reference_circles.keys()) if reference_circles else 0
            else:
                ref_max = 0
            max_radius_for_rays = max(max_radius_for_rays, ref_max)

        # Process ray_label_radius parameter
        default_label_radius = max_radius_for_rays * 1.1
        if ray_label_radius is None:
            # Default: same radius for all rays
            label_radii_dict = {num: default_label_radius for num in ray_numbers}
        elif isinstance(ray_label_radius, dict):
            # Dictionary mapping ray numbers to radii
            label_radii_dict = {}
            for ray_num in ray_numbers:
                label_radii_dict[ray_num] = ray_label_radius.get(ray_num, default_label_radius)
        elif isinstance(ray_label_radius, (list, tuple, np.ndarray)):
            # List of radii (must match number of displayed rays)
            if len(ray_label_radius) != len(ray_numbers):
                raise ValueError(f"Length of ray_label_radius list ({len(ray_label_radius)}) must match number of displayed rays ({len(ray_numbers)})")
            label_radii_dict = {ray_numbers[i]: ray_label_radius[i] for i in range(len(ray_numbers))}
        elif isinstance(ray_label_radius, (int, float)):
            # Single radius for all rays
            label_radii_dict = {num: float(ray_label_radius) for num in ray_numbers}
        else:
            raise ValueError(f"ray_label_radius must be float, dict, or list, got {type(ray_label_radius)}")

        # Draw rays and labels
        for i, (direction, ray_number) in enumerate(zip(ray_dirs_normalized, ray_numbers)):
            # Get label radius for this specific ray
            label_radius_for_ray = label_radii_dict[ray_number]

            # Draw ray line from origin to slightly beyond the label
            ray_endpoint = direction * label_radius_for_ray * 1.05
            ax.plot([0, ray_endpoint[0]], [0, ray_endpoint[1]],
                   color=ray_color, linewidth=ray_linewidth, alpha=ray_alpha, zorder=1)

            # Place label at specified radius for this ray
            label_pos = direction * label_radius_for_ray
            ax.text(label_pos[0], label_pos[1], str(ray_number),
                   fontsize=ray_label_fontsize, ha='center', va='center',
                   bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', edgecolor=ray_color, alpha=0.8),
                   zorder=5)

        # Update max_radius to include all label radii for axis limits
        max_label_radius = max(label_radii_dict.values()) if label_radii_dict else default_label_radius

    # Determine colors for each label
    if label_colors is not None:
        # Use explicitly provided colors
        if isinstance(label_colors, dict):
            # Dictionary mapping labels to colors
            colors = [label_colors.get(label, 'gray') for label in unique_labels]
        elif isinstance(label_colors, (list, tuple)):
            # List of colors 
            if len(label_colors) != len(unique_labels):
                raise ValueError(f"Length of label_colors ({len(label_colors)}) must match number of unique labels ({len(unique_labels)})")
            colors = list(label_colors)
        else:
            raise ValueError(f"label_colors must be list or dict, got {type(label_colors)}")
    else: 
        # Generate colors from colormap
        cmap = plt.get_cmap(colormap)
        colors = [cmap(i / len(unique_labels)) for i in range(len(unique_labels))]


    # Plot each label group with a different color
    for i, label in enumerate(unique_labels):
        mask = labels_array == label
        ax.scatter(scaled_vectors[mask, 0], scaled_vectors[mask, 1],
                  s=point_size, alpha=point_alpha, c=[colors[i]],
                  edgecolors='black', linewidth=0.5, label=f'{label}', zorder=3)

    # Show axes through origin
    if show_axes:
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)

    # Set equal aspect ratio and limits based on max radius
    ax.set_aspect('equal')
    max_radius = max(radii_dict.values()) if radii_dict else 1.0

    # Also consider reference circles when setting limits
    if reference_circles is not None:
        if isinstance(reference_circles, (list, tuple)):
            ref_max = max(reference_circles) if reference_circles else 0
        elif isinstance(reference_circles, dict):
            ref_max = max(reference_circles.keys()) if reference_circles else 0
        else:
            ref_max = 0
        max_radius = max(max_radius, ref_max)

    # Also consider ray label radius when setting limits
    if ray_directions is not None:
        max_radius = max(max_radius, max_label_radius)

    axis_limit = max_radius * 1.2
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)

    # Labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)

    # Grid
    if show_grid:
        ax.grid(True, alpha=0.3)

    # Legend
    if show_legend:
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

    plt.tight_layout()

    return fig, ax






def plot_2d_scatter_with_bins(
    data_tensor,
    index_tensor,
    label_list,
    bins=None,
    bins_mode='numbers',
    label_colors=None,
    figsize=(12, 8),
    title='All Distributions Combined',
    point_size=25,
    point_alpha=0.7,
    preserve_aspect=False,
    show_legend=True,
    show_grid=True
):
    """
    Create a 2D scatter plot with labeled data points and optional bin overlay.

    This replicates the "All Distributions Combined" view from SimpleDistribution2DSampler.

    Parameters:
    -----------
    data_tensor : torch.Tensor
        n x 2 tensor where each row is a 2D data point (x, y coordinates)
    index_tensor : torch.Tensor
        1D tensor of length n with label indices for each point
    label_list : list
        List of label names corresponding to the indices
    bins : torch.Tensor, optional
        m x 2 tensor of bin center coordinates to overlay on the plot.
        If None, no bins are shown.
    bins_mode : str, optional
        How to display bins: 'numbers' (numbered annotations) or 'x' (X markers).
        Default: 'numbers'
    label_colors : list or dict, optional
        Colors for each label. Can be:
        - List of colors (same order as label_list)
        - Dict mapping label names to colors
        If None, uses tab10 colormap (matching SimpleDistribution2DSampler)
    figsize : tuple
        Figure size (width, height). Default: (12, 8)
    title : str
        Plot title. Default: 'All Distributions Combined'
    point_size : float
        Size of scatter points. Default: 25
    point_alpha : float
        Transparency of points (0-1). Default: 0.7
    preserve_aspect : bool
        Whether to use equal aspect ratio. Default: False
    show_legend : bool
        Whether to show legend. Default: True
    show_grid : bool
        Whether to show grid. Default: True

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects

    Example:
    --------
    >>> # From sampler widget
    >>> settings = sampler.get_all_settings(include_data=True)
    >>> data = settings['data']['pytorch_data']
    >>> entropy_data = settings['data']['entropy_data']
    >>>
    >>> fig, ax = plot_2d_scatter_with_bins(
    ...     data_tensor=data['data_tensor'],
    ...     index_tensor=data['index_tensor'],
    ...     label_list=data['label_list'],
    ...     bins=entropy_data['prob_dist_bins__no_heads'],
    ...     bins_mode='numbers',
    ...     label_colors=settings['data']['colors']
    ... )
    """
    # Validate inputs
    if data_tensor.dim() != 2 or data_tensor.shape[1] != 2:
        raise ValueError("data_tensor must be n x 2 (each row is a 2D point)")

    if len(index_tensor) != len(data_tensor):
        raise ValueError(f"index_tensor length ({len(index_tensor)}) must match data_tensor rows ({len(data_tensor)})")

    # Convert to numpy
    data_np = data_tensor.numpy()
    indices_np = index_tensor.numpy()

    # Generate colors if not provided (match SimpleDistribution2DSampler formula)
    n_labels = len(label_list)
    if label_colors is None:
        colors_mpl = plt.cm.tab10(np.linspace(0, 1, n_labels))
        colors = [colors_mpl[i] for i in range(n_labels)]
    elif isinstance(label_colors, dict):
        colors = [label_colors.get(label, 'gray') for label in label_list]
    else:
        colors = list(label_colors)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot each label group with its color
    for i, label in enumerate(label_list):
        mask = indices_np == i
        ax.scatter(
            data_np[mask, 0],
            data_np[mask, 1],
            alpha=point_alpha,
            s=point_size,
            color=colors[i],
            label=label,
            edgecolors='black',
            linewidth=0.3
        )

    # Plot bins if provided
    if bins is not None:
        if bins.dim() != 2 or bins.shape[1] != 2:
            raise ValueError("bins must be m x 2 (each row is a 2D bin center)")

        bins_np = bins.numpy()

        if bins_mode == 'x':
            # Plot as X markers
            ax.scatter(
                bins_np[:, 0],
                bins_np[:, 1],
                alpha=0.8,
                s=80,
                color='black',
                marker='x',
                label='Prob Dist Bins',
                linewidths=2
            )
        elif bins_mode == 'numbers':
            # Plot as numbered text annotations (starting at 1)
            for i, (x, y) in enumerate(bins_np):
                ax.annotate(
                    str(i + 1),
                    (x, y),
                    fontsize=8,
                    color='black',
                    weight='bold',
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle='circle,pad=0.2', facecolor='white', alpha=0.8)
                )
            # Add a legend entry for the numbers
            ax.scatter([], [], alpha=0, label='Prob Dist Bins (numbered)')
        else:
            raise ValueError(f"bins_mode must be 'x' or 'numbers', got '{bins_mode}'")

    # Configure plot
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)

    if show_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    if show_grid:
        ax.grid(True, alpha=0.3)

    if preserve_aspect:
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    return fig, ax







def plot_1d_scatter_labeled(
    data_tensor,
    index_tensor,
    label_list,
    bins=None,
    label_colors=None,
    label_y_positions=None,
    label_y_spacing=0.3,
    ylabel_colors=None,
    bin_label_y_spacing=None,
    figsize=(12, 6),
    title='1D Distribution Scatter Plot',
    point_size=30,
    point_alpha=0.7,
    colormap='tab10',
    show_legend=True,
    show_grid=True,
    show_distribution_lines=True,
    xlabel='Value',
    ylabel='Distribution',
    bin_line_color='gray',
    bin_line_alpha=0.5,
    bin_line_width=1.5,
    bin_label_fontsize=10,
    bin_label_y_offset=0.3
):
    """
    Create a 1D scatter plot with labeled data points on horizontal lines.

    Each distribution is plotted on its own horizontal line, with optional
    vertical lines marking bin boundaries/centers.

    Parameters:
    -----------
    data_tensor : torch.Tensor
        n x 1 tensor where each row is a 1D data point
    index_tensor : torch.Tensor
        1D tensor of length n with label indices for each point
    label_list : list
        List of label names corresponding to the indices
    bins : torch.Tensor or array-like, optional
        1D tensor/array of bin positions to mark with vertical lines.
        If None, no bins are shown.
    label_colors : list or dict, optional
        Colors for each label. Can be:
        - List of colors (same order as label_list)
        - Dict mapping label names to colors
        If None, uses colormap (default: tab10)
    label_y_positions : dict or list, optional
        Y-axis positions for each label. Can be:
        - Dict mapping label names to y-positions
        - List of y-positions (same order as label_list)
        If None, distributes labels evenly using label_y_spacing
    label_y_spacing : float, optional
        Vertical spacing between distribution lines when label_y_positions is None.
        Default: 0.3
    ylabel_colors : list or dict, optional
        Colors for y-axis tick labels (population labels). Can be:
        - List of colors (same order as label_list)
        - Dict mapping label names to colors
        If None, uses default black color for all labels
    bin_label_y_spacing : float, optional
        Vertical spacing between bin label levels and from top distribution to bins.
        If None, uses label_y_spacing (same as distribution spacing).
        Set to a smaller value for tighter bin label spacing.
        Default: None (uses label_y_spacing)
    figsize : tuple
        Figure size (width, height). Default: (12, 6)
    title : str
        Plot title
    point_size : float
        Size of scatter points. Default: 30
    point_alpha : float
        Transparency of points (0-1). Default: 0.7
    colormap : str
        Matplotlib colormap name. Default: 'tab10'
    show_legend : bool
        Whether to show legend. Default: True
    show_grid : bool
        Whether to show grid. Default: True
    show_distribution_lines : bool
        Whether to draw horizontal lines for each distribution. Default: True
    xlabel : str
        Label for x-axis. Default: 'Value'
    bin_line_color : str
        Color for bin marker lines. Default: 'gray'
    bin_line_alpha : float
        Transparency of bin lines (0-1). Default: 0.5
    bin_line_width : float
        Width of bin lines. Default: 1.5
    bin_label_fontsize : int
        Font size for bin number labels. Default: 8
    bin_label_y_offset : float
        Vertical offset for bin labels from top of plot. Default: 0.3

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects

    Example:
    --------
    >>> # Extract 1D projection (e.g., x-coordinate only)
    >>> data_1d = data_tensor[:, 0:1]  # Keep as n x 1
    >>>
    >>> # Extract bin x-coordinates
    >>> bins_1d = bins[:, 0] if bins is not None else None
    >>>
    >>> fig, ax = plot_1d_scatter_labeled(
    ...     data_tensor=data_1d,
    ...     index_tensor=index_tensor,
    ...     label_list=label_list,
    ...     bins=bins_1d,
    ...     label_colors=colors
    ... )
    """
    # Validate inputs
    if data_tensor.dim() == 1:
        data_tensor = data_tensor.unsqueeze(1)  # Convert n -> n x 1

    if data_tensor.dim() != 2 or data_tensor.shape[1] != 1:
        raise ValueError("data_tensor must be n x 1 (each row is a 1D point)")

    if len(index_tensor) != len(data_tensor):
        raise ValueError(f"index_tensor length ({len(index_tensor)}) must match data_tensor rows ({len(data_tensor)})")

    # Convert to numpy
    data_np = data_tensor.numpy().flatten()  # n x 1 -> n
    indices_np = index_tensor.numpy()

    # Generate colors if not provided
    n_labels = len(label_list)
    if label_colors is None:
        colors_mpl = plt.cm.get_cmap(colormap)
        colors = [colors_mpl(i / n_labels) for i in range(n_labels)]
    elif isinstance(label_colors, dict):
        colors = [label_colors.get(label, 'gray') for label in label_list]
    else:
        colors = list(label_colors)

    # Determine y-positions for each label (in DESCENDING order vertically)
    if label_y_positions is None:
        # Default: evenly spaced with configurable spacing, descending order
        y_positions_dict = {label: (n_labels - 1 - i) * label_y_spacing for i, label in enumerate(label_list)}
    elif isinstance(label_y_positions, dict):
        y_positions_dict = label_y_positions
    elif isinstance(label_y_positions, (list, tuple)):
        if len(label_y_positions) != n_labels:
            raise ValueError(f"Length of label_y_positions ({len(label_y_positions)}) must match number of labels ({n_labels})")
        y_positions_dict = {label: label_y_positions[i] for i, label in enumerate(label_list)}
    else:
        raise ValueError(f"label_y_positions must be dict or list, got {type(label_y_positions)}")

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Get x-axis range for horizontal lines
    x_min = data_np.min() if len(data_np) > 0 else 0
    x_max = data_np.max() if len(data_np) > 0 else 1
    x_padding = (x_max - x_min) * 0.05 if x_max > x_min else 0.1

    # Plot each label group on its horizontal line
    for i, label in enumerate(label_list):
        mask = indices_np == i
        y_pos = y_positions_dict[label]

        # Draw horizontal line for this distribution (if enabled)
        if show_distribution_lines:
            ax.plot([x_min - x_padding, x_max + x_padding], [y_pos, y_pos],
                    color='k', linewidth=0.5, alpha=0.3, zorder=1)

        # Create y-values (all at same height for this label)
        y_values = np.full(np.sum(mask), y_pos)

        ax.scatter(
            data_np[mask],
            y_values,
            alpha=point_alpha,
            s=point_size,
            color=colors[i],
            label=label,
            edgecolors='black',
            linewidth=0.3,
            zorder=3
        )

    # Plot bin markers if provided
    if bins is not None:
        # Convert to numpy
        if isinstance(bins, torch.Tensor):
            bins_np = bins.numpy()
        else:
            bins_np = np.array(bins)

        if bins_np.ndim != 1:
            raise ValueError(f"bins must be 1D, got shape {bins_np.shape}")

        # Use bin_label_y_spacing if provided, otherwise use label_y_spacing
        bin_spacing = bin_label_y_spacing if bin_label_y_spacing is not None else label_y_spacing

        # Determine y-range for bin lines with uniform spacing
        # Bottom margin should be label_y_spacing below the lowest distribution
        y_min = min(y_positions_dict.values()) - label_y_spacing
        # Set y_max so that the lowest bin label is bin_spacing above the top distribution
        # Since label positions are computed as: y_max + label_offset (where min offset is bin_label_y_offset)
        # We want: y_max + bin_label_y_offset = max(y_positions) + bin_spacing
        # Therefore: y_max = max(y_positions) + bin_spacing - bin_label_y_offset
        y_max = max(y_positions_dict.values()) + bin_spacing - bin_label_y_offset

        # Sort bins by position to detect overlaps
        sorted_indices = np.argsort(bins_np)

        # Compute label y-offsets to avoid overlaps with UNIFORM vertical spacing
        # Estimate label width in data units (rough approximation based on data range)
        x_range = np.ptp(data_np) if len(data_np) > 0 else 1.0
        # Assume each label character takes ~0.02 of data range, with ~2 digits average
        min_spacing = max(0.04 * x_range, 0.1)  # At least 0.1 in data units

        label_y_offsets = {}
        current_level = 0
        prev_bin_pos = None

        for idx in sorted_indices:
            bin_pos = bins_np[idx]

            if prev_bin_pos is not None:
                spacing = abs(bin_pos - prev_bin_pos)
                if spacing < min_spacing:
                    # Too close - move to next level
                    current_level = (current_level + 1) % 3  # Use 3 levels
                else:
                    # Far enough - reset to base level
                    current_level = 0

            # Use uniform spacing: each level is exactly bin_spacing apart
            # Level 0: bin_label_y_offset
            # Level 1: bin_label_y_offset + bin_spacing
            # Level 2: bin_label_y_offset + 2 * bin_spacing
            label_y_offsets[idx] = bin_label_y_offset + current_level * bin_spacing
            prev_bin_pos = bin_pos

        # Find max offset for y-axis limits
        max_offset = max(label_y_offsets.values()) if label_y_offsets else bin_label_y_offset

        # Draw vertical lines for each bin
        for i, bin_pos in enumerate(bins_np):
            # Compute label position (center of circle)
            label_offset = label_y_offsets[i]
            label_y_pos = y_max + label_offset

            # Draw vertical line extending to the center of the circle (not beyond)
            # The line goes from y_min to label_y_pos (where the circle center is)
            ax.plot([bin_pos, bin_pos], [y_min, label_y_pos],
                    color=bin_line_color, linewidth=bin_line_width,
                    alpha=bin_line_alpha, zorder=1)

            # Add bin number label at top with computed offset
            # Match plot_unit_circle_scatter_labeled style: bbox with circle pad and edgecolor
            # Text is always black to match unit circle function
            ax.text(
                bin_pos,
                label_y_pos,
                str(i + 1),  # Start counting at 1
                fontsize=bin_label_fontsize,
                ha='center',
                va='center',  # Center alignment to match unit circle style
                color='black',  # Always black to match unit circle style
                weight='bold',
                bbox=dict(boxstyle='circle,pad=0.3', facecolor='white',
                         edgecolor=bin_line_color, alpha=0.8),
                zorder=5  # Matches unit circle style
            )

    # Configure plot
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    # Set y-axis ticks to label names
    y_ticks = [y_positions_dict[label] for label in label_list]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(label_list)

    # Color the y-axis tick labels if ylabel_colors is provided
    if ylabel_colors is not None:
        if isinstance(ylabel_colors, dict):
            # Dictionary mapping label names to colors
            for tick_label, label in zip(ax.get_yticklabels(), label_list):
                color = ylabel_colors.get(label, 'black')
                tick_label.set_color(color)
        elif isinstance(ylabel_colors, (list, tuple)):
            # List of colors (same order as label_list)
            if len(ylabel_colors) != n_labels:
                raise ValueError(f"Length of ylabel_colors ({len(ylabel_colors)}) must match number of labels ({n_labels})")
            for tick_label, color in zip(ax.get_yticklabels(), ylabel_colors):
                tick_label.set_color(color)
        else:
            raise ValueError(f"ylabel_colors must be list or dict, got {type(ylabel_colors)}")

    # Set y-axis limits with uniform spacing
    if bins is not None:
        # Bottom margin: exactly label_y_spacing below the lowest distribution
        y_min_limit = min(y_positions_dict.values()) - label_y_spacing
        # Compute the highest label position (circle center)
        highest_label_y_pos = y_max + max_offset
        # Top margin: use bin_spacing for consistency with bin label spacing
        # bin_spacing was set earlier in the bins section
        y_max_limit = highest_label_y_pos + bin_spacing
        ax.set_ylim(y_min_limit, y_max_limit)
    else:
        # No bins - use label_y_spacing for margins
        y_min_limit = min(y_positions_dict.values()) - label_y_spacing
        y_max_limit = max(y_positions_dict.values()) + label_y_spacing
        ax.set_ylim(y_min_limit, y_max_limit)

    if show_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    if show_grid:
        ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    return fig, ax
