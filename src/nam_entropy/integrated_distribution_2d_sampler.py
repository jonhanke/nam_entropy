import torch
import torch.distributions as dist
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual, HBox, VBox, Output
from IPython.display import display, clear_output
import seaborn as sns
import hashlib
import pandas as pd
import os
from datetime import datetime

# Try to import the data prep functions and entropy calculation, but provide fallbacks
try:
    from .data_prep import prepare_labeled_tensor_dataset, convert_tensor_list_to_dataframe
    DATA_PREP_AVAILABLE = True
except ImportError:
    print("Warning: Could not import data prep functions. Using built-in fallbacks.")
    DATA_PREP_AVAILABLE = False

try:
    import sys
    sys.path.append('../src/nam_entropy/')
    from h import compute_all_entropy_measures
    ENTROPY_AVAILABLE = True
#    print("✓ Entropy calculation functions imported successfully")
except ImportError:
    print("Warning: Could not import entropy calculation functions. Entropy analysis will be disabled.")
    ENTROPY_AVAILABLE = False


## ===================================================================
## ==================  Fallback Data Prep Functions ================
## ===================================================================

def fallback_prepare_labeled_tensor_dataset(data_tensor_list, input_tensor_labels_list):
    """Fallback implementation if external functions not available"""
    all_data = []
    all_labels = []

    for tensor, label in zip(data_tensor_list, input_tensor_labels_list):
        n_samples = tensor.shape[0]
        all_data.append(tensor)
        all_labels.extend([label] * n_samples)

    data_tensor = torch.cat(all_data, dim=0)

    unique_labels = sorted(list(set(input_tensor_labels_list)))
    label_to_index = {label: i for i, label in enumerate(unique_labels)}

    index_tensor = torch.tensor([label_to_index[label] for label in all_labels])

    return index_tensor, data_tensor, unique_labels, label_to_index

def fallback_convert_tensor_list_to_dataframe(data_tensor_list, input_tensor_labels_list, save_path=None):
    """Fallback implementation to create DataFrame and save CSV"""
    all_data = []
    all_labels = []

    for tensor, label in zip(data_tensor_list, input_tensor_labels_list):
        # Convert to numpy
        if isinstance(tensor, torch.Tensor):
            numpy_array = tensor.detach().cpu().numpy()
        else:
            numpy_array = tensor

        n_samples = numpy_array.shape[0]
        all_data.append(numpy_array)
        all_labels.extend([label] * n_samples)

    # Stack all data
    stacked_data = np.vstack(all_data)

    # Create DataFrame
    df = pd.DataFrame({
        'feature_0': stacked_data[:, 0],
        'feature_1': stacked_data[:, 1],
        'label': all_labels
    })

    # Save to CSV
    if save_path is not None:
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f"✓ DataFrame saved to: {save_path}")
            print(f"  File size: {os.path.getsize(save_path)} bytes")
        except Exception as e:
            print(f"✗ Error saving CSV: {e}")

    return df


## ===================================================================
## ==================  2D Distribution Functions ====================
## ===================================================================

def create_2d_uniform_rectangle(x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0):
    """Create a 2D uniform distribution over a rectangle"""
    low = torch.tensor([x_min, y_min])
    high = torch.tensor([x_max, y_max])
    return dist.Uniform(low, high)

def create_2d_gaussian(x_mean=0.0, y_mean=0.0, x_std=1.0, y_std=1.0, correlation=0.0):
    """Create a 2D Gaussian distribution with optional correlation"""
    # Create covariance matrix
    cov_xy = correlation * x_std * y_std
    cov_matrix = torch.tensor([[x_std**2, cov_xy],
                              [cov_xy, y_std**2]])
    mean_vector = torch.tensor([x_mean, y_mean])
    return dist.MultivariateNormal(mean_vector, cov_matrix)


## ===========================================================
## ==================  2D Sample Cache ====================
## ===========================================================

# Global cache for 2D samples (up to 10 distributions)
SAMPLE_2D_CACHE = {}
for i in range(1, 11):
    SAMPLE_2D_CACHE[f'dist{i}_samples'] = None
    SAMPLE_2D_CACHE[f'dist{i}_hash'] = None

def get_2d_param_hash(dist_type, **params):
    """Create a hash of distribution type and parameters for caching"""
    param_str = f"{dist_type}_{sorted(params.items())}"
    return hashlib.md5(param_str.encode()).hexdigest()[:8]

def clear_unused_cache_entries(n_active_distributions):
    """Clear cache entries for distributions not currently active"""
    global SAMPLE_2D_CACHE
    for i in range(n_active_distributions + 1, 11):
        SAMPLE_2D_CACHE[f'dist{i}_samples'] = None
        SAMPLE_2D_CACHE[f'dist{i}_hash'] = None

def get_active_cache_data(n_distributions):
    """Get only the cache data for active distributions"""
    active_data = {}
    for i in range(1, n_distributions + 1):
        if SAMPLE_2D_CACHE[f'dist{i}_samples'] is not None:
            active_data[f'dist{i}_samples'] = SAMPLE_2D_CACHE[f'dist{i}_samples']
            active_data[f'dist{i}_hash'] = SAMPLE_2D_CACHE[f'dist{i}_hash']
    return active_data


## ===================================================================
## ==================  Integrated2DDistributionWidget Class ====================
## ===================================================================

class Integrated2DDistributionWidget:
    def __init__(self, save_path=None, max_distributions=10):
        # Set default save path to today's daily folder
        if save_path is None:
            today = datetime.now().strftime('%Y-%m-%d')
            daily_dir = f".code_dev/{today}"
            self.save_path = f"{daily_dir}/sample_2d_distribution_data.csv"
        else:
            self.save_path = save_path

        self.max_distributions = max_distributions
        self.output = Output(layout=widgets.Layout(min_height='600px'))  # Fixed minimum height
        self.param_container = None  # Container for dynamic parameter widgets
        self.entropy_display = Output(layout=widgets.Layout(min_height='150px'))  # Fixed entropy display height
        self.entropy_section = None  # Will be created in display() - the container we'll show/hide
        self.latest_dataframe = None  # Store the latest generated DataFrame
        self._last_pytorch_data = None  # Store PyTorch tensor data
        self._last_entropy_data = None  # Store entropy calculation results
        self._last_entropy_error = None  # Store entropy calculation errors
        self.create_widgets()
        self.setup_observers()

    def get_data(self):
        """Public method to get the latest generated data"""
        return self.latest_dataframe

    def get_cache_data(self):
        """Public method to get active cache data"""
        n_dists = self.n_distributions.value
        return get_active_cache_data(n_dists)

    def get_entropy_data(self):
        """Public method to get entropy calculation results"""
        return self._last_entropy_data

    def get_all_settings(self, include_data=False):
        """Get complete current configuration and optionally include data

        Args:
            include_data (bool): If True, includes DataFrame, cache, PyTorch tensor data, and entropy results

        Returns:
            dict: Complete settings and optionally data
        """
        settings = {
            'widget_settings': {
                'n_distributions': self.n_distributions.value,
                'n_samples': self.n_samples.value,
                'show_individual_plots': self.show_individual_plots.value,
                'show_combined_plot': self.show_combined_plot.value,
                'show_entropy_display': self.show_entropy_display.value,
                'combined_plot_size': self.combined_plot_size.value,
                'preserve_aspect_ratio': self.preserve_aspect_ratio.value,
                'show_prob_dist_bins': self.show_prob_dist_bins.value,
                'bins_display_mode': self.bins_display_mode.value,
                'n_bins': self.n_bins.value
            },
            'file_settings': {
                'save_path': self.save_path,
                'max_distributions': self.max_distributions
            },
            'distributions': {}
        }

        # Get distribution configurations for active distributions
        n_dists = self.n_distributions.value
        for i in range(1, n_dists + 1):
            dist_key = f'dist{i}'
            dist_type = self.dist_types[dist_key].value
            params = self.get_distribution_params(dist_type, dist_key)
            settings['distributions'][f'dist{i}'] = {
                'type': dist_type,
                'params': params
            }

        # Optionally include data
        if include_data:
            settings['data'] = {
                'dataframe': self.get_data(),
                'cache_data': self.get_cache_data(),
                'pytorch_data': self._last_pytorch_data,
                'entropy_data': self._last_entropy_data,
                'dataframe_info': {
                    'shape': self.latest_dataframe.shape if self.latest_dataframe is not None else None,
                    'columns': list(self.latest_dataframe.columns) if self.latest_dataframe is not None else None,
                    'sample_count_per_distribution': self.n_samples.value,
                    'total_samples': n_dists * self.n_samples.value
                }
            }

        return settings

    def create_widgets(self):
        """Create all widgets for 2D distribution sampling"""

        # Number of distributions selector
        self.n_distributions = widgets.IntSlider(
            min=1, max=self.max_distributions, step=1, value=2,
            description='# Distributions:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='450px')
        )

        # Sample count
        self.n_samples = widgets.IntSlider(
            min=100, max=5000, step=100, value=1000,
            description='# Samples:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='450px')
        )

        # Number of bins for entropy calculation
        self.n_bins = widgets.IntSlider(
            min=5, max=50, step=1, value=20,
            description='# Bins:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='350px')
        )

        # Show individual plots toggle
        self.show_individual_plots = widgets.Checkbox(
            value=True,
            description='Show Individual Plots',
            style={'description_width': '200px'},
            layout=widgets.Layout(width='400px'),
            indent=False
        )

        # Show combined plot toggle
        self.show_combined_plot = widgets.Checkbox(
            value=True,
            description='Show Combined Plot',
            style={'description_width': '200px'},
            layout=widgets.Layout(width='400px'),
            indent=False
        )

        # Show entropy display toggle
        self.show_entropy_display = widgets.Checkbox(
            value=True,
            description='Show Entropy Display',
            style={'description_width': '200px'},
            layout=widgets.Layout(width='400px'),
            indent=False
        )

        # Combined plot size options
        self.combined_plot_size = widgets.Dropdown(
            options=[('Normal', 'normal'), ('Large', 'large'), ('Extra Large', 'xlarge')],
            value='large',
            description='Combined Plot Size:',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='350px')
        )

        # Preserve aspect ratio option
        self.preserve_aspect_ratio = widgets.Checkbox(
            value=False,
            description='Equal Aspect Ratio',
            style={'description_width': '200px'},
            layout=widgets.Layout(width='400px'),
            indent=False
        )

        # Show probability distribution bins toggle
        self.show_prob_dist_bins = widgets.Checkbox(
            value=True,
            description='Show Prob Dist Bins',
            style={'description_width': '200px'},
            layout=widgets.Layout(width='400px'),
            indent=False
        )

        # Show probability distribution bar chart toggle
        self.show_prob_dist_chart = widgets.Checkbox(
            value=True,
            description='Show Prob Dist Chart',
            style={'description_width': '200px'},
            layout=widgets.Layout(width='400px'),
            indent=False
        )

        # Show label entropies toggle
        self.show_label_entropies = widgets.Checkbox(
            value=True,
            description='Show Label Entropies',
            style={'description_width': '200px'},
            layout=widgets.Layout(width='400px'),
            indent=False
        )

        # Bins display mode (numbers vs X markers)
        self.bins_display_mode = widgets.Dropdown(
            options=[('X Markers', 'x'), ('Numbers', 'numbers')],
            value='numbers',
            description='Bins Display:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='300px')
        )

        # Distribution type selectors (create for all possible distributions)
        self.dist_types = {}
        default_types = ['Uniform Rectangle', 'Gaussian'] + ['Gaussian'] * (self.max_distributions - 2)

        for i in range(1, self.max_distributions + 1):
            self.dist_types[f'dist{i}'] = widgets.Dropdown(
                options=['Uniform Rectangle', 'Gaussian'],
                value=default_types[i-1] if i <= len(default_types) else 'Gaussian',
                description=f'Dist {i} Type:',
                style={'description_width': '100px'},
                layout=widgets.Layout(width='280px')
            )

        # Create parameter widgets for all distributions
        self.uniform_params = {}
        self.gaussian_params = {}

        # Default values for each distribution
        uniform_defaults = [
            (-1, 1, -1, 1),    # Dist 1
            (0, 2, 0, 2),      # Dist 2
            (-0.5, 0.5, 1, 2), # Dist 3
        ] + [(-1, 1, -1, 1)] * (self.max_distributions - 3)  # Rest use default

        gaussian_defaults = [
            (0, 0, 1, 1, 0),      # Dist 1: (x_mean, y_mean, x_std, y_std, corr)
            (1, 1, 0.8, 0.8, 0.3), # Dist 2
            (-0.5, -0.5, 0.6, 0.6, -0.2), # Dist 3
        ] + [(0, 0, 1, 1, 0)] * (self.max_distributions - 3)  # Rest use default

        for i in range(1, self.max_distributions + 1):
            # Uniform parameters with wider layout
            u_defaults = uniform_defaults[i-1] if i <= len(uniform_defaults) else uniform_defaults[-1]
            self.uniform_params[f'dist{i}'] = {
                'x_min': widgets.FloatSlider(min=-3, max=2, step=0.1, value=u_defaults[0],
                                           description=f'D{i} x_min:', style={'description_width': '80px'},
                                           layout=widgets.Layout(width='250px')),
                'x_max': widgets.FloatSlider(min=-2, max=3, step=0.1, value=u_defaults[1],
                                           description=f'D{i} x_max:', style={'description_width': '80px'},
                                           layout=widgets.Layout(width='250px')),
                'y_min': widgets.FloatSlider(min=-3, max=2, step=0.1, value=u_defaults[2],
                                           description=f'D{i} y_min:', style={'description_width': '80px'},
                                           layout=widgets.Layout(width='250px')),
                'y_max': widgets.FloatSlider(min=-2, max=3, step=0.1, value=u_defaults[3],
                                           description=f'D{i} y_max:', style={'description_width': '80px'},
                                           layout=widgets.Layout(width='250px'))
            }

            # Gaussian parameters with wider layout
            g_defaults = gaussian_defaults[i-1] if i <= len(gaussian_defaults) else gaussian_defaults[-1]
            self.gaussian_params[f'dist{i}'] = {
                'x_mean': widgets.FloatSlider(min=-2, max=2, step=0.1, value=g_defaults[0],
                                            description=f'D{i} x_mean:', style={'description_width': '80px'},
                                            layout=widgets.Layout(width='250px')),
                'y_mean': widgets.FloatSlider(min=-2, max=2, step=0.1, value=g_defaults[1],
                                            description=f'D{i} y_mean:', style={'description_width': '80px'},
                                            layout=widgets.Layout(width='250px')),
                'x_std': widgets.FloatSlider(min=0.1, max=2, step=0.1, value=g_defaults[2],
                                           description=f'D{i} x_std:', style={'description_width': '80px'},
                                           layout=widgets.Layout(width='250px')),
                'y_std': widgets.FloatSlider(min=0.1, max=2, step=0.1, value=g_defaults[3],
                                           description=f'D{i} y_std:', style={'description_width': '80px'},
                                           layout=widgets.Layout(width='250px')),
                'correlation': widgets.FloatSlider(min=-0.9, max=0.9, step=0.1, value=g_defaults[4],
                                                 description=f'D{i} corr:', style={'description_width': '80px'},
                                                 layout=widgets.Layout(width='250px'))
            }

    def setup_observers(self):
        """Set up event observers for all widgets"""
        # Number of distributions observer - this will update the parameter display
        self.n_distributions.observe(self.update_parameter_display, names='value')

        # Display-only options observers - these only update display, not data
        self.show_individual_plots.observe(self.update_display_only, names='value')
        self.show_combined_plot.observe(self.update_display_only, names='value')
        self.show_entropy_display.observe(self.update_entropy_display_visibility, names='value')
        self.show_label_entropies.observe(lambda change: self.update_entropy_display(), names='value')
        self.combined_plot_size.observe(self.update_display_only, names='value')
        self.preserve_aspect_ratio.observe(self.update_display_only, names='value')
        self.show_prob_dist_bins.observe(self.update_display_only, names='value')
        self.show_prob_dist_chart.observe(self.update_display_only, names='value')
        self.bins_display_mode.observe(self.update_display_only, names='value')

        # Distribution type observers - these will rebuild the parameter display
        for dist_widget in self.dist_types.values():
            dist_widget.observe(self.update_parameter_display, names='value')

        # Sample count and bins observer - these affect data generation
        self.n_samples.observe(self.update_plot, names='value')
        self.n_bins.observe(self.update_plot, names='value')

        # Parameter observers for all distributions - these affect data generation
        for dist_params in self.uniform_params.values():
            for widget in dist_params.values():
                widget.observe(self.update_plot, names='value')

        for dist_params in self.gaussian_params.values():
            for widget in dist_params.values():
                widget.observe(self.update_plot, names='value')

    def update_display_only(self, change=None):
        """Update only the plot display without regenerating data or recalculating entropy"""
        # Don't clear output - just update the existing plot to prevent jumping
        n_dists = self.n_distributions.value
        show_individual = self.show_individual_plots.value
        show_combined = self.show_combined_plot.value
        plot_size = self.combined_plot_size.value
        preserve_aspect = self.preserve_aspect_ratio.value
        show_prob_bins = self.show_prob_dist_bins.value
        show_prob_chart = self.show_prob_dist_chart.value
        bins_mode = self.bins_display_mode.value

        # Check if we have cached data to plot
        if not any(SAMPLE_2D_CACHE[f'dist{i}_samples'] is not None for i in range(1, n_dists + 1)):
            return

        # Get distribution info from cache (no regeneration)
        dist_info = []
        for i in range(1, n_dists + 1):
            cache_key = f'dist{i}'
            if SAMPLE_2D_CACHE[f'{cache_key}_samples'] is not None:
                dist_key = f'dist{i}'
                dist_type = self.dist_types[dist_key].value
                params = self.get_distribution_params(dist_type, dist_key)

                dist_info.append({
                    'type': dist_type,
                    'params': params,
                    'samples': SAMPLE_2D_CACHE[f'{cache_key}_samples'],
                    'index': i
                })

        # Clear and render plots with cached data
        with self.output:
            clear_output(wait=True)
            self.render_plots(dist_info, show_individual, show_combined, plot_size, preserve_aspect, show_prob_bins, show_prob_chart, bins_mode)

    def update_entropy_display_visibility(self, change=None):
        """Update the entire entropy section visibility when checkbox changes"""
        if self.entropy_section is not None:
            # Toggle visibility using layout.display
            if self.show_entropy_display.value:
                self.entropy_section.layout.visibility = 'visible'
                self.entropy_section.layout.display = 'flex'
                # Update the content
                self.update_entropy_display()
            else:
                self.entropy_section.layout.visibility = 'hidden'
                self.entropy_section.layout.display = 'none'

    def get_distribution_params(self, dist_type, dist_key):
        """Extract relevant parameters for a distribution type"""
        if dist_type == 'Uniform Rectangle':
            uniform_params = self.uniform_params[dist_key]
            return {
                'x_min': uniform_params['x_min'].value,
                'x_max': uniform_params['x_max'].value,
                'y_min': uniform_params['y_min'].value,
                'y_max': uniform_params['y_max'].value
            }
        elif dist_type == 'Gaussian':
            gaussian_params = self.gaussian_params[dist_key]
            return {
                'x_mean': gaussian_params['x_mean'].value,
                'y_mean': gaussian_params['y_mean'].value,
                'x_std': gaussian_params['x_std'].value,
                'y_std': gaussian_params['y_std'].value,
                'correlation': gaussian_params['correlation'].value
            }

    def create_distribution(self, dist_type, params):
        """Create a distribution object from type and parameters"""
        if dist_type == 'Uniform Rectangle':
            return create_2d_uniform_rectangle(**params)
        elif dist_type == 'Gaussian':
            return create_2d_gaussian(**params)

    def build_parameter_widgets(self):
        """Build parameter widgets for the current number of active distributions"""
        n_dists = self.n_distributions.value

        # Build parameter controls for active distributions only
        dist_columns = []
        cols_per_row = min(3, n_dists)  # Max 3 columns

        # Use consistent colors matching the scatter plots
        colors = plt.cm.tab10(np.linspace(0, 1, n_dists))

        for i in range(1, n_dists + 1):
            dist_key = f'dist{i}'
            dist_type = self.dist_types[dist_key].value

            # Distribution type selector
            type_selector = self.dist_types[dist_key]

            # Get the appropriate parameter widgets based on selected distribution type
            if dist_type == 'Uniform Rectangle':
                param_widgets = list(self.uniform_params[dist_key].values())
                param_label = "<b>Uniform Rectangle Parameters:</b>"
            else:  # Gaussian
                param_widgets = list(self.gaussian_params[dist_key].values())
                param_label = "<b>Gaussian Parameters:</b>"

            # Get color for this distribution (convert matplotlib color to hex)
            color_rgba = colors[i - 1]
            color_hex = '#{:02x}{:02x}{:02x}'.format(int(color_rgba[0]*255), int(color_rgba[1]*255), int(color_rgba[2]*255))

            # Column for this distribution - only show relevant parameters with wider layout
            dist_column = VBox([
                widgets.HTML(value=f"<h4 style='text-align: center; color: {color_hex};'>Distribution {i}</h4>"),
                type_selector,
                widgets.HTML(value=f"<div style='text-align: center;'>{param_label}</div>"),
                VBox(param_widgets)
            ], layout=widgets.Layout(width=f'{95//cols_per_row}%', margin='5px', min_width='300px', align_items='center'))

            dist_columns.append(dist_column)

        # Arrange columns in rows
        param_rows = []
        for i in range(0, len(dist_columns), cols_per_row):
            row_columns = dist_columns[i:i + cols_per_row]
            param_rows.append(HBox(row_columns, layout=widgets.Layout(justify_content='space-around')))

        return VBox(param_rows)

    def update_parameter_display(self, change=None):
        """Update the parameter display when number of distributions changes or distribution type changes"""
        # Clear unused cache entries when number of distributions changes
        if change and change['name'] == 'value' and hasattr(change['owner'], 'description'):
            if '# Distributions' in change['owner'].description:
                clear_unused_cache_entries(change['new'])

        # Clear and rebuild the parameter container
        new_params = self.build_parameter_widgets()

        if self.param_container is not None:
            # Replace children of the container
            self.param_container.children = new_params.children
        else:
            self.param_container = new_params

        # Update the plot with full data regeneration
        self.update_plot()

    def update_entropy_display(self):
        """Update the entropy metrics display with enhanced visual layout"""
        with self.entropy_display:
            clear_output(wait=False)

            # Always show content when this is called (visibility controlled by entropy_section)
            if self._last_entropy_data is not None and ENTROPY_AVAILABLE:
                output_dict = self._last_entropy_data['output_dict']

                # Extract entropy metrics from the 'output_metrics' key - handle both tensor and scalar values
                entropy_metrics = {}
                output_metrics = output_dict.get('output_metrics', {})
                if 'entropy' in output_metrics:
                    entropy_val = output_metrics['entropy']
                    entropy_metrics['entropy'] = entropy_val.item() if hasattr(entropy_val, 'item') else float(entropy_val)
                if 'conditional_entropy' in output_metrics:
                    cond_entropy_val = output_metrics['conditional_entropy']
                    entropy_metrics['conditional_entropy'] = cond_entropy_val.item() if hasattr(cond_entropy_val, 'item') else float(cond_entropy_val)
                if 'mutual_information' in output_metrics:
                    mutual_info_val = output_metrics['mutual_information']
                    entropy_metrics['mutual_information'] = mutual_info_val.item() if hasattr(mutual_info_val, 'item') else float(mutual_info_val)

                # Create and display the entropy metric boxes
                if entropy_metrics:
                    # Update box values
                    entropy_val = entropy_metrics.get('entropy', None)
                    cond_entropy_val = entropy_metrics.get('conditional_entropy', None)
                    mutual_info_val = entropy_metrics.get('mutual_information', None)

                    # Entropy H(X) box
                    entropy_display_val = f"{entropy_val:.4f}" if entropy_val is not None else "---"
                    entropy_box = widgets.HTML(
                        value=f"<div style='text-align: center;'><h4 style='color: #2E86C1; margin: 0;'>Entropy H(Z)</h4><p style='font-size: 18px; margin: 5px 0; font-weight: bold;'>{entropy_display_val}</p></div>",
                        layout=widgets.Layout(width='30%', border='2px solid #2E86C1', padding='15px', margin='5px')
                    )

                    # Conditional Entropy H(X|Y) box
                    cond_entropy_display_val = f"{cond_entropy_val:.4f}" if cond_entropy_val is not None else "---"
                    cond_entropy_box = widgets.HTML(
                        value=f"<div style='text-align: center;'><h4 style='color: #E74C3C; margin: 0;'>Conditional Entropy H(Z|L)</h4><p style='font-size: 18px; margin: 5px 0; font-weight: bold;'>{cond_entropy_display_val}</p></div>",
                        layout=widgets.Layout(width='30%', border='2px solid #E74C3C', padding='15px', margin='5px')
                    )

                    # Mutual Information I(X;Y) box
                    mutual_info_display_val = f"{mutual_info_val:.4f}" if mutual_info_val is not None else "---"
                    mutual_info_box = widgets.HTML(
                        value=f"<div style='text-align: center;'><h4 style='color: #28B463; margin: 0;'>Mutual Information I(Z,L)</h4><p style='font-size: 18px; margin: 5px 0; font-weight: bold;'>{mutual_info_display_val}</p></div>",
                        layout=widgets.Layout(width='30%', border='2px solid #28B463', padding='15px', margin='5px')
                    )

                    # Display the boxes
                    metric_boxes = HBox([entropy_box, cond_entropy_box, mutual_info_box],
                                       layout=widgets.Layout(justify_content='space-around', margin='10px 0'))
                    display(metric_boxes)


                    # Display label-dependent conditional entropies (second row)
                    if self.show_label_entropies.value:
                        label_entropy_dict = output_metrics.get('label_entropy_dict', {})
                        if label_entropy_dict:

                            # Filter out 'total_population' to get only label-specific entropies
                            label_entropies_raw = {k: v for k, v in label_entropy_dict.items() if k != 'total_population'}

                            # Get number of distributions
                            n_dists = len(label_entropies_raw)

                            # Shorten "Distribution" to "Dist" when there are more than 5 distributions
                            label_entropies = {}
                            for k, v in label_entropies_raw.items():
                                if (n_dists < 8) and (n_dists > 5):
                                    shortened_label = k.replace('Distribution ', 'Dist ')
                                    label_entropies[shortened_label] = v
                                elif n_dists >= 8:
                                    shortened_label = k.replace('Distribution ', '')
                                    label_entropies[shortened_label] = v
                                else:
                                    label_entropies[k] = v

                            # Sort labels by numeric index to ensure correct order (1, 2, ..., 9, 10)
                            def extract_number(label):
                                import re
                                match = re.search(r'(\d+)', label)
                                return int(match.group(1)) if match else 0

                            label_entropies = dict(sorted(label_entropies.items(), key=lambda x: extract_number(x[0])))


                            if label_entropies:
                                # Create boxes for each label entropy
                                label_boxes = []
                                # Use consistent colors matching the scatter plots
                                colors_mpl = plt.cm.tab10(np.linspace(0, 1, n_dists))

                                for idx, (label, entropy_value) in enumerate(label_entropies.items()):
                                    color_rgba = colors_mpl[idx]
                                    color_hex = '#{:02x}{:02x}{:02x}'.format(int(color_rgba[0]*255), int(color_rgba[1]*255), int(color_rgba[2]*255))
                                    entropy_display_val = f"{entropy_value:.4f}" if entropy_value is not None else "---"
                                    label_box = widgets.HTML(
                                        value=f"<div style='text-align: center;'><h4 style='color: {color_hex}; margin: 0; font-size: 14px;'>H(Z|L={label})</h4><p style='font-size: 16px; margin: 5px 0; font-weight: bold;'>{entropy_display_val}</p></div>",
                                        layout=widgets.Layout(width=f'{max(15, 90//len(label_entropies))}%', border=f'2px solid {color_hex}', padding='10px', margin='5px')
                                    )
                                    label_boxes.append(label_box)

                                # Display label entropy boxes
                                label_boxes_container = HBox(label_boxes,
                                                            layout=widgets.Layout(justify_content='space-around', margin='10px 0'))
                                display(label_boxes_container)


                else:
                    # Fallback display for no metrics
                    fallback_box = widgets.HTML(
                        value="<div style='text-align: center; padding: 20px; border: 2px solid #FFA500; background-color: #FFF3CD;'><h4 style='color: #856404; margin: 0;'>⚠️ Entropy metrics not available in output</h4></div>",
                        layout=widgets.Layout(width='100%', margin='10px 0')
                    )
                    display(fallback_box)
            else:
                if not ENTROPY_AVAILABLE:
                    # Error message box
                    error_box = widgets.HTML(
                        value="<div style='text-align: center; padding: 20px; border: 2px solid #DC3545; background-color: #F8D7DA;'><h4 style='color: #721C24; margin: 0;'>❌ Entropy calculations disabled (import failed)</h4></div>",
                        layout=widgets.Layout(width='100%', margin='10px 0')
                    )
                    display(error_box)
                else:
                    # Check if there's an error to display
                    if hasattr(self, '_last_entropy_error') and self._last_entropy_error is not None:
                        error_box = widgets.HTML(
                            value=f"<div style='text-align: center; padding: 20px; border: 2px solid #DC3545; background-color: #F8D7DA;'><h4 style='color: #721C24; margin: 0;'>❌ Entropy Calculation Failed</h4><pre style='text-align: left; font-size: 10px; margin: 10px 0; overflow: auto;'>{self._last_entropy_error}</pre></div>",
                            layout=widgets.Layout(width='100%', margin='10px 0')
                        )
                        display(error_box)
                    else:
                        waiting_box = widgets.HTML(
                            value="<div style='text-align: center; padding: 20px; border: 2px solid #6C757D; background-color: #E2E3E5;'><h4 style='color: #383D41; margin: 0;'>⏳ No entropy data available yet</h4></div>",
                            layout=widgets.Layout(width='100%', margin='10px 0')
                        )
                        display(waiting_box)

    def get_figure_dimensions(self, n_dists, plot_size, show_individual, show_combined, preserve_aspect):
        """Calculate optimal figure dimensions based on settings"""

        # Base dimensions for different plot sizes
        size_configs = {
            'normal': {'base_width': 12, 'base_height': 8},
            'large': {'base_width': 16, 'base_height': 10},
            'xlarge': {'base_width': 20, 'base_height': 12}
        }

        config = size_configs[plot_size]

        if not show_individual and show_combined:
            # Combined plot only - adjust for aspect ratio
            if preserve_aspect:
                # Make it larger when aspect ratio is preserved
                return (config['base_width'] * 1.2, config['base_height'] * 1.2)
            else:
                return (config['base_width'], config['base_height'])
        elif show_individual and not show_combined:
            # Individual plots only
            if n_dists <= 5:
                cols = n_dists
                individual_rows = 1
            else:
                cols = 5
                individual_rows = (n_dists + cols - 1) // cols

            width = config['base_width']
            total_height = individual_rows * 4  # Just individual plots

            return (width, total_height)
        elif show_individual and show_combined:
            # Both individual and combined plots
            if n_dists <= 5:
                cols = n_dists
                individual_rows = 1
            else:
                cols = 5
                individual_rows = (n_dists + cols - 1) // cols

            # Use full width and calculate height based on content
            width = config['base_width']

            # Calculate height: give more space to combined plot, especially with aspect ratio
            individual_plot_height = 2.5  # Smaller individual plots

            if preserve_aspect:
                # Give even more space to combined plot when aspect ratio is preserved
                combined_plot_height = 8
            else:
                combined_plot_height = 6

            total_height = individual_rows * individual_plot_height + combined_plot_height

            return (width, total_height)
        else:
            # Neither individual nor combined - just return minimal size
            return (config['base_width'] * 0.5, config['base_height'] * 0.5)

    def render_plots(self, dist_info, show_individual, show_combined, plot_size, preserve_aspect, show_prob_bins, show_prob_chart, bins_mode='x'):
        """Render plots with given distribution info"""
        n_dists = len(dist_info)
        figsize = self.get_figure_dimensions(n_dists, plot_size, show_individual, show_combined, preserve_aspect)

        # Close any existing matplotlib figures to prevent accumulation
        plt.close('all')

        # Handle case where neither plot type is selected
        if not show_individual and not show_combined:
            print("No plots selected for display. Please enable either individual plots or combined plot.")
            return

        # Determine plot layout based on options
        if not show_individual and show_combined:
            # Combined plot only
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            fig.suptitle(f'Integrated 2D Distribution Sampler - Combined View ({n_dists} Distributions)', fontsize=16)

            # Combined plot
            colors = plt.cm.tab10(np.linspace(0, 1, n_dists))

            for i, info in enumerate(dist_info):
                samples = info['samples']
                dist_type = info['type']

                # Create detailed label with distribution info
                if dist_type == 'Uniform Rectangle':
                    params = info['params']
                    label = f"Dist {info['index']}: Uniform [{params['x_min']:.1f},{params['x_max']:.1f}]×[{params['y_min']:.1f},{params['y_max']:.1f}]"
                else:  # Gaussian
                    params = info['params']
                    label = f"Dist {info['index']}: Gaussian μ=({params['x_mean']:.1f},{params['y_mean']:.1f}), σ=({params['x_std']:.1f},{params['y_std']:.1f})"

                # Use larger scatter points for combined-only view
                scatter_size = 30 if preserve_aspect else 20
                ax.scatter(samples[:, 0].numpy(), samples[:, 1].numpy(),
                         alpha=0.7, s=scatter_size, color=colors[i], label=label)

            # Plot probability distribution bins if available and requested
            if show_prob_bins and self._last_entropy_data is not None:
                prob_bins = self._last_entropy_data['prob_dist_bins__no_heads']
                if prob_bins is not None:
                    self._plot_bins(ax, prob_bins, bins_mode)

            ax.set_title('All Distributions Combined', fontsize=14)
            ax.set_xlabel('X', fontsize=12)
            ax.set_ylabel('Y', fontsize=12)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)

            # Set aspect ratio if requested
            if preserve_aspect:
                ax.set_aspect('equal', adjustable='box')

            plt.show()

        elif show_individual and not show_combined:
            # Individual plots only
            if n_dists <= 5:
                cols = n_dists
                rows = 1
            else:
                cols = 5
                rows = (n_dists + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=figsize)
            if rows == 1 and cols == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()

            fig.suptitle(f'Integrated 2D Distribution Sampler - Individual Plots ({n_dists} Distributions)', fontsize=16)

            # Colors for each distribution
            colors = plt.cm.tab10(np.linspace(0, 1, n_dists))

            # Plot individual distributions
            for i, info in enumerate(dist_info):
                ax = axes[i]
                samples = info['samples']
                dist_type = info['type']
                params = info['params']

                # Individual scatter plot
                ax.scatter(samples[:, 0].numpy(), samples[:, 1].numpy(),
                          alpha=0.6, s=15, color=colors[i])
                ax.set_title(f'Dist {info["index"]}: {dist_type}', fontsize=12)
                ax.set_xlabel('X', fontsize=10)
                ax.set_ylabel('Y', fontsize=10)
                ax.grid(True, alpha=0.3)

                # Set aspect ratio if requested
                if preserve_aspect:
                    ax.set_aspect('equal', adjustable='box')

                # Add parameter text
                if dist_type == 'Uniform Rectangle':
                    param_text = f"x∈[{params['x_min']:.1f}, {params['x_max']:.1f}]\ny∈[{params['y_min']:.1f}, {params['y_max']:.1f}]"
                else:  # Gaussian
                    param_text = f"μ=({params['x_mean']:.1f}, {params['y_mean']:.1f})\nσ=({params['x_std']:.1f}, {params['y_std']:.1f})\nρ={params['correlation']:.1f}"

                ax.text(0.02, 0.98, param_text, transform=ax.transAxes,
                       verticalalignment='top', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

            # Hide unused subplots
            for i in range(n_dists, len(axes)):
                axes[i].set_visible(False)

            plt.show()

        else:
            # Show both individual and combined plots with improved layout
            if n_dists <= 3:
                # For 1-3 distributions: horizontal layout
                cols = n_dists
                individual_rows = 1

                # Adjust height ratios based on aspect ratio setting
                if preserve_aspect:
                    height_ratios = [1, 3.2]  # Give even more space to combined plot
                else:
                    height_ratios = [1, 2.4]

                # Create a 2x(n_dists) grid with custom height ratios
                fig = plt.figure(figsize=figsize, constrained_layout=True)
                gs = fig.add_gridspec(2, cols, height_ratios=height_ratios)
                fig.suptitle(f'Integrated 2D Distribution Sampler ({n_dists} Distributions)', fontsize=16)

                # Colors for each distribution
                colors = plt.cm.tab10(np.linspace(0, 1, n_dists))

                # Plot individual distributions in top row
                for i, info in enumerate(dist_info):
                    ax = fig.add_subplot(gs[0, i])

                    samples = info['samples']
                    dist_type = info['type']
                    params = info['params']

                    # Individual scatter plot
                    ax.scatter(samples[:, 0].numpy(), samples[:, 1].numpy(),
                              alpha=0.6, s=8, color=colors[i])
                    ax.set_title(f'Dist {info["index"]}: {dist_type}', fontsize=10)
                    ax.set_xlabel('X', fontsize=9)
                    ax.set_ylabel('Y', fontsize=9)
                    ax.grid(True, alpha=0.3)

                    # Set aspect ratio if requested
                    if preserve_aspect:
                        ax.set_aspect('equal', adjustable='box')

                    # Add parameter text
                    if dist_type == 'Uniform Rectangle':
                        param_text = f"x∈[{params['x_min']:.1f}, {params['x_max']:.1f}]\ny∈[{params['y_min']:.1f}, {params['y_max']:.1f}]"
                    else:  # Gaussian
                        param_text = f"μ=({params['x_mean']:.1f}, {params['y_mean']:.1f})\nσ=({params['x_std']:.1f}, {params['y_std']:.1f})\nρ={params['correlation']:.1f}"

                    ax.text(0.02, 0.98, param_text, transform=ax.transAxes,
                           verticalalignment='top', fontsize=6,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

                # Combined plot spanning entire bottom row
                combined_ax = fig.add_subplot(gs[1, :])

            else:
                # For 4+ distributions: use grid layout for individual plots
                cols = min(5, n_dists)
                individual_rows = (n_dists + cols - 1) // cols

                # Adjust height ratios based on aspect ratio setting
                if preserve_aspect:
                    combined_height_ratio = 3.5  # Give even more space to combined plot
                else:
                    combined_height_ratio = 2.5

                # Create subplot layout with emphasis on combined plot
                height_ratios = [1] * individual_rows + [combined_height_ratio]
                rows = individual_rows + 1

                fig = plt.figure(figsize=figsize, constrained_layout=True)
                gs = fig.add_gridspec(rows, cols, height_ratios=height_ratios)
                fig.suptitle(f'Integrated 2D Distribution Sampler ({n_dists} Distributions)', fontsize=16)

                # Colors for each distribution
                colors = plt.cm.tab10(np.linspace(0, 1, n_dists))

                # Plot each distribution individually
                for i, info in enumerate(dist_info):
                    row = i // cols
                    col = i % cols
                    ax = fig.add_subplot(gs[row, col])

                    samples = info['samples']
                    dist_type = info['type']
                    params = info['params']

                    # Individual scatter plot
                    ax.scatter(samples[:, 0].numpy(), samples[:, 1].numpy(),
                              alpha=0.6, s=8, color=colors[i])
                    ax.set_title(f'Dist {info["index"]}: {dist_type}', fontsize=10)
                    ax.set_xlabel('X', fontsize=9)
                    ax.set_ylabel('Y', fontsize=9)
                    ax.grid(True, alpha=0.3)

                    # Set aspect ratio if requested
                    if preserve_aspect:
                        ax.set_aspect('equal', adjustable='box')

                    # Add parameter text
                    if dist_type == 'Uniform Rectangle':
                        param_text = f"x∈[{params['x_min']:.1f}, {params['x_max']:.1f}]\ny∈[{params['y_min']:.1f}, {params['y_max']:.1f}]"
                    else:  # Gaussian
                        param_text = f"μ=({params['x_mean']:.1f}, {params['y_mean']:.1f})\nσ=({params['x_std']:.1f}, {params['y_std']:.1f})\nρ={params['correlation']:.1f}"

                    ax.text(0.02, 0.98, param_text, transform=ax.transAxes,
                           verticalalignment='top', fontsize=6,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

                # Combined plot spanning multiple columns in the last row
                combined_ax = fig.add_subplot(gs[-1, :])

            # Create the combined plot (same for both layouts)
            for i, info in enumerate(dist_info):
                samples = info['samples']
                dist_type = info['type']

                # Create concise label for combined plot
                label = f"Dist {info['index']} ({dist_type})"

                # Use larger scatter points when aspect ratio is preserved
                scatter_size = 35 if preserve_aspect else 25
                combined_ax.scatter(samples[:, 0].numpy(), samples[:, 1].numpy(),
                                  alpha=0.7, s=scatter_size, color=colors[i], label=label)

            # Plot probability distribution bins if available and requested
            if show_prob_bins and self._last_entropy_data is not None:
                prob_bins = self._last_entropy_data['prob_dist_bins__no_heads']
                if prob_bins is not None:
                    self._plot_bins(combined_ax, prob_bins, bins_mode)

            combined_ax.set_title('All Distributions Combined', fontsize=14)
            combined_ax.set_xlabel('X', fontsize=12)
            combined_ax.set_ylabel('Y', fontsize=12)
            combined_ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
            combined_ax.grid(True, alpha=0.3)

            # Set aspect ratio if requested
            if preserve_aspect:
                combined_ax.set_aspect('equal', adjustable='box')

            plt.show()

        # Plot probability distribution bar chart if requested and data available
        if show_prob_chart and self._last_entropy_data is not None and ENTROPY_AVAILABLE:
            try:
                prob_dist_by_label_tensor = self._last_entropy_data['prob_dist_by_label_tensor']
                prob_dist_for_total_population_tensor = self._last_entropy_data['prob_dist_for_total_population_tensor']
                label_list = self._last_pytorch_data['label_list']

                # Create display tensor and labels (same as in Notebook A)
                display_label_list = label_list + ['Total population']
                display_tensor = torch.cat([prob_dist_by_label_tensor, prob_dist_for_total_population_tensor.unsqueeze(0)], dim=0)

                # Plot the bar chart
                self._plot_prob_dist_bars(display_tensor, display_label_list)
            except Exception as e:
                pass  # Silently skip if there's an error

    def _plot_prob_dist_bars(self, tensor_data, labels):
        """Plot soft-binned probability distributions as overlaid bar chart"""
        num_rows = len(tensor_data)
        n_bins = tensor_data.shape[1]
        n_dists = self.n_distributions.value

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Use consistent colors matching the scatter plots - same as in render_plots
        colors = plt.cm.tab10(np.linspace(0, 1, n_dists))

        # Extend colors array to include gray for "Total population"
        colors_extended = list(colors) + [np.array([0.5, 0.5, 0.5, 1.0])]  # Gray for total population

        # Calculate bar width and positions
        bar_width = 0.8 / num_rows
        x = np.arange(n_bins)

        # Plot each distribution
        for i, (row, label) in enumerate(zip(tensor_data, labels)):
            values = row.numpy()
            offset = (i - num_rows/2 + 0.5) * bar_width
            ax.bar(x + offset, values, bar_width, label=label, alpha=0.7, color=colors_extended[i])

        ax.set_xlabel('Bin Number', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title('Soft-binned Probability Distributions by Label', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(x + 1)  # Start bin numbering at 1
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _plot_bins(self, ax, prob_bins, bins_mode):
        """Plot probability distribution bins on the given axis"""
        if bins_mode == 'x':
            # Plot as X markers
            ax.scatter(prob_bins[:, 0].numpy(), prob_bins[:, 1].numpy(),
                     alpha=0.8, s=80, color='black', marker='x',
                     label='Prob Dist Bins', linewidths=2)
        elif bins_mode == 'numbers':
            # Plot as numbered text annotations (starting at 1)
            for i, (x, y) in enumerate(prob_bins.numpy()):
                ax.annotate(str(i + 1), (x, y), fontsize=8, color='black',
                          weight='bold', ha='center', va='center',
                          bbox=dict(boxstyle='circle,pad=0.2', facecolor='white', alpha=0.8))
            # Add a legend entry for the numbers
            ax.scatter([], [], alpha=0, label='Prob Dist Bins (numbered)')

    def update_plot(self, change=None):
        """Update the plot with current parameters - includes data regeneration and entropy calculation"""
        global SAMPLE_2D_CACHE

        with self.output:
            clear_output(wait=True)

            n_dists = self.n_distributions.value
            show_individual = self.show_individual_plots.value
            show_combined = self.show_combined_plot.value
            plot_size = self.combined_plot_size.value
            preserve_aspect = self.preserve_aspect_ratio.value
            show_prob_bins = self.show_prob_dist_bins.value
            show_prob_chart = self.show_prob_dist_chart.value
            bins_mode = self.bins_display_mode.value

            # Clear unused cache entries first
            clear_unused_cache_entries(n_dists)

            # Get current distribution types and parameters for active distributions
            dist_info = []
            for i in range(1, n_dists + 1):
                dist_key = f'dist{i}'
                dist_type = self.dist_types[dist_key].value
                params = self.get_distribution_params(dist_type, dist_key)
                distribution = self.create_distribution(dist_type, params)

                # Generate cache hash including n_bins
                dist_hash = get_2d_param_hash(dist_type, **params, n_samples=self.n_samples.value, n_bins=self.n_bins.value)

                # Generate samples only if parameters changed
                cache_key = f'dist{i}'
                if SAMPLE_2D_CACHE[f'{cache_key}_hash'] != dist_hash:
                    SAMPLE_2D_CACHE[f'{cache_key}_samples'] = distribution.sample((self.n_samples.value,))
                    SAMPLE_2D_CACHE[f'{cache_key}_hash'] = dist_hash
                    # print(f"Regenerated samples for Distribution {i} ({dist_type})")  # Suppressed

                dist_info.append({
                    'type': dist_type,
                    'params': params,
                    'samples': SAMPLE_2D_CACHE[f'{cache_key}_samples'],
                    'index': i
                })

            ## DATA PREP FOR ANALYSIS:
            ## -----------------------

            # Prepare data for analysis
            samples_tensors_list = [info['samples'] for info in dist_info]
            samples_labels_list = [f"Distribution {info['index']}" for info in dist_info]

            # Prepare labeled dataset and save CSV
            try:
                if DATA_PREP_AVAILABLE:
                    index_tensor, data_tensor, label_list, label_to_label_index_dict = \
                        prepare_labeled_tensor_dataset(samples_tensors_list, input_tensor_labels_list=samples_labels_list)

                    # Save as dataframe (suppress output)
                    import sys
                    from io import StringIO
                    old_stdout = sys.stdout
                    sys.stdout = StringIO()
                    self.latest_dataframe = convert_tensor_list_to_dataframe(
                        samples_tensors_list,
                        input_tensor_labels_list=samples_labels_list,
                        save_path=self.save_path
                    )
                    sys.stdout = old_stdout
                else:
                    index_tensor, data_tensor, label_list, label_to_label_index_dict = \
                        fallback_prepare_labeled_tensor_dataset(samples_tensors_list, samples_labels_list)

                    # Save as dataframe using fallback (suppress output)
                    import sys
                    from io import StringIO
                    old_stdout = sys.stdout
                    sys.stdout = StringIO()
                    self.latest_dataframe = fallback_convert_tensor_list_to_dataframe(
                        samples_tensors_list,
                        samples_labels_list,
                        save_path=self.save_path
                    )
                    sys.stdout = old_stdout

                # Store PyTorch tensor data for get_all_settings
                self._last_pytorch_data = {
                    'index_tensor': index_tensor,
                    'data_tensor': data_tensor,
                    'label_list': label_list,
                    'label_to_label_index_dict': label_to_label_index_dict,
                    'samples_tensors_list': samples_tensors_list,
                    'samples_labels_list': samples_labels_list
                }

                ## ENTROPY CALCULATIONS:
                ## ---------------------

                if ENTROPY_AVAILABLE:
                    try:
                        # Compute the entropies using the integrated code pattern with specified number of bins
                        output_dict = compute_all_entropy_measures(
                            data_embeddings_tensor=data_tensor,
                            data_label_indices_tensor=index_tensor,
                            label_list=label_list,
                            n_bins=self.n_bins.value
                        )

                        # Extract the intermediate data from the entropy calculation
                        intermediate_data_dict = output_dict['intermediate_data']
                        prob_dist_for_total_population_tensor = intermediate_data_dict['prob_dist_for_total_population_tensor']
                        prob_dist_by_label_tensor = intermediate_data_dict['prob_dist_by_label_tensor']

                        prob_dist_bins = output_dict['intermediate_data']['tmp_bins']  ## Remove the heads
                        prob_dist_bins__no_heads = prob_dist_bins.squeeze(1)

                        # Store entropy data
                        self._last_entropy_data = {
                            'output_dict': output_dict,
                            'intermediate_data_dict': intermediate_data_dict,
                            'prob_dist_bins__no_heads': prob_dist_bins__no_heads,
                            'prob_dist_for_total_population_tensor': prob_dist_for_total_population_tensor,
                            'prob_dist_by_label_tensor': prob_dist_by_label_tensor
                        }

                        # print(f"✓ Entropy calculations completed with {self.n_bins.value} bins")  # Suppressed
                        # print(f"  Prob dist bins shape: {prob_dist_bins__no_heads.shape}")  # Suppressed

                    except Exception as e:
                        import traceback
                        error_msg = f"⚠ Entropy calculation error: {e}\n{traceback.format_exc()}"
                        print(error_msg)
                        self._last_entropy_error = error_msg  # Store for display
                        self._last_entropy_data = None
                else:
                    self._last_entropy_data = None

                # print(f"Data shape: {self.latest_dataframe.shape} | Active distributions: {n_dists}")  # Suppressed

            except Exception as e:
                print(f"Data prep error: {e}")
                self.latest_dataframe = None
                self._last_pytorch_data = None
                self._last_entropy_data = None

            # Update entropy display
            self.update_entropy_display()

            # Render the plots
            self.render_plots(dist_info, show_individual, show_combined, plot_size, preserve_aspect, show_prob_bins, show_prob_chart, bins_mode)

    def display(self):
        """Display the complete interface"""
        # Controls header
        entropy_status = "✓ Enabled" if ENTROPY_AVAILABLE else "✗ Disabled"
        controls_header = VBox([
            widgets.HTML(value=f"<h3 style='text-align: center; color: #333;'>Integrated 2D Distribution Sampler with Entropy Analysis</h3>"),
        #    widgets.HTML(value=f"<p style='text-align: center; color: #666;'>Entropy calculations: {entropy_status}</p>")
        ])

        # Main controls - horizontal layout for distribution count and samples
        main_controls = HBox([
            self.n_distributions,
            widgets.HTML(value="<div style='width: 30px;'></div>"),
            self.n_samples,
            widgets.HTML(value="<div style='width: 30px;'></div>"),
            self.n_bins
        ], layout=widgets.Layout(justify_content='center', margin='10px 0'))

        # Plot options - vertical layout for better space utilization with wider widgets
        plot_options_left = VBox([
            self.show_individual_plots,
            self.show_combined_plot,
            self.show_entropy_display,
            self.show_label_entropies,            
            self.preserve_aspect_ratio,
            self.show_prob_dist_bins,
            self.show_prob_dist_chart
        ], layout=widgets.Layout(flex='1 1 auto'))

        plot_options_right = VBox([
            self.combined_plot_size,
            self.bins_display_mode
        ], layout=widgets.Layout(flex='1 1 auto'))

        plot_options = HBox([
            plot_options_left,
            plot_options_right
        ], layout=widgets.Layout(justify_content='space-around', margin='10px 0', width='100%'))

        # Entropy display section - this is the container we'll show/hide
        self.entropy_section = VBox([
            widgets.HTML(value="<h4 style='text-align: center; color: #333; margin: 10px 0;'>Entropy Metrics</h4>"),
            self.entropy_display
        ], layout=widgets.Layout(margin='10px 0'))

        ## Initialize entropy section visibility based on checkbox state
        self.update_entropy_display_visibility()
  
        # Build initial parameter container
        self.param_container = self.build_parameter_widgets()

        # Complete interface - keep the same output widget
        main_interface = VBox([
            controls_header,
            main_controls,
            plot_options,
            self.entropy_section,
            self.param_container,
            self.output
        ])

        display(main_interface)

        # Initial plot
        self.update_plot()



## ===================================================================
## ====================  User Display Routines =======================
## ===================================================================

def SimpleDistribution2DSampler(save_path=None, max_distributions=10):
    """Factory function providing Distribution2DSampler interface for backward compatibility.

    Creates an Integrated2DDistributionWidget with entropy features hidden by default,
    matching the simpler interface used in Notebook B.

    Args:
        save_path: Path to save CSV data (optional)
        max_distributions: Maximum number of distributions (default: 10)

    Returns:
        Integrated2DDistributionWidget instance configured for simple sampling mode
    """
    # Create the integrated widget
    widget = Integrated2DDistributionWidget(save_path=save_path, max_distributions=max_distributions)

    # Configure for simple mode (hide entropy-related features by default)
    widget.show_entropy_display.value = False
    widget.show_prob_dist_bins.value = False
    widget.show_prob_dist_chart.value = False
    widget.show_combined_plot.value = True  # Always show combined plot in simple mode

    return widget



def Distribution2DSampler(save_path=None, max_distributions=10):
    """Factory function providing Distribution2DSampler interface for backward compatibility.

    Creates an Integrated2DDistributionWidget with entropy features hidden by default,
    matching the simpler interface used in Notebook B.

    Args:
        save_path: Path to save CSV data (optional)
        max_distributions: Maximum number of distributions (default: 10)

    Returns:
        Integrated2DDistributionWidget instance configured for simple sampling mode
    """
    # Create the integrated widget
    widget = Integrated2DDistributionWidget(save_path=save_path, max_distributions=max_distributions)

    # Configure for simple mode (hide entropy-related features by default)
    widget.show_entropy_display.value = True
    widget.show_prob_dist_bins.value = True
    widget.show_prob_dist_chart.value = True
    widget.show_combined_plot.value = True  # Always show combined plot in simple mode

    return widget