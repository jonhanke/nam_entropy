


import torch
import matplotlib.pyplot as plt

def plot_tensor_bars(tensor_data, figsize=(10, 6), title='Bar Chart from Tensor', 
                     xlabel='Index', ylabel='Value', show_grid=True, separate_plots=True, labels=None):
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
    """
    # Handle both 1D and 2D tensors
    if tensor_data.dim() == 1:
        # Single row - create one bar chart
        values = tensor_data.numpy()
        plt.figure(figsize=figsize)
        plt.bar(range(len(values)), values)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(range(len(values)))
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
                axes[i].bar(range(len(values)), values)
                axes[i].set_title(f'{title} - {labels[i]}')
                axes[i].set_xlabel(xlabel)
                axes[i].set_ylabel(ylabel)
                axes[i].set_xticks(range(len(values)))
                if show_grid:
                    axes[i].grid(axis='y', alpha=0.3)
        else:
            # Plot all rows on the same chart
            plt.figure(figsize=figsize)
            width = 0.8 / num_rows  # Divide bar width among rows
            
            for i, row in enumerate(tensor_data):
                values = row.numpy()
                x_pos = [x + (i - num_rows/2 + 0.5) * width for x in range(len(values))]
                plt.bar(x_pos, values, width=width, label=labels[i], alpha=0.8)
            
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.xticks(range(len(tensor_data[0])))
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

