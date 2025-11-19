
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Literal
import yaml



class DatasetConfig(BaseModel):
    """Configuration for dataset loading with support for multiple sources."""
    
    # Data source options (specify ONLY ONE)
    task_repo: Optional[str] = 'hcoxec/french_german_mix'
    subtask_name: Optional[str] = None
    data_path: Optional[str] = None
    data_url: Optional[str] = None
    
    # Dataset structure
    data_split_name: str = 'train'
    data_column_name_list: List[str] = Field(default_factory=lambda: ['sentence'])
    label_column_name: str = 'language'
    
    # File format (for local/remote files)
    file_format: Optional[Literal['csv', 'json', 'jsonl', 'parquet', 'arrow', 'txt']] = None
    
    @model_validator(mode='after')
    def validate_data_source(self):
        """Ensure exactly one data source is specified."""
        sources = [
            self.task_repo is not None,
            self.data_path is not None,
            self.data_url is not None
        ]
        
        if sum(sources) > 1:
            raise ValueError(
                "Specify only ONE data source: task_repo, data_path, or data_url"
            )
        
        # At least one source must be specified
        if sum(sources) == 0:
            raise ValueError(
                "Must specify at least one data source"
            )
        
        return self
    
    def get_source_type(self) -> Literal['huggingface', 'local', 'url']:
        """Determine which data source is being used."""
        if self.task_repo is not None:
            return 'huggingface'
        elif self.data_path is not None:
            return 'local'
        elif self.data_url is not None:
            return 'url'


class ModelConfig(BaseModel):
    """Configuration for model loading and setup."""
    
    model_id: str = 'distilbert/distilbert-base-multilingual-cased'
    output_hidden_states: bool = True
    device_map: str = 'auto'
    cache_dir: str = 'model_cache'


class TokenizerConfig(BaseModel):
    """Configuration for tokenizer settings."""
    
    max_len: int = Field(default=256, gt=0, description="Maximum token length")
    truncation: bool = True


class TrainingConfig(BaseModel):
    """Configuration for training/run parameters."""
    
    seed: int = Field(default=496, ge=0)
    n_batches: int = Field(default=10, gt=0, description="How many batches of data to analyze")
    batch_size: int = Field(default=256, gt=0, description="How many sentences per batch")
    device: str = 'cpu'
    saved_results_path: str = 'results'



class EntropyEstimatorConfig(BaseModel):
    """Configuration for entropy estimator/soft-binning parameters."""
    
    # Binning/estimation parameters
    n_bins: int = Field(default=1000, gt=0, description="Number of bins for histogram")
    n_heads: int = Field(default=1, gt=0, description="Number of attention heads")
    bin_type: Literal['unit_sphere', 'uniform'] = Field(
        default='uniform',
        description="Type of binning: 'unit_sphere' or 'uniform'"
    )
    dist_fn: Literal['cosine', 'euclidean'] = Field(
        default='euclidean',
        description="Distance function for binning"
    )
    smoothing_fn: Literal['softmax', 'None'] = Field(
        default='None',
        description="Smoothing function to apply"
    )
    smoothing_temp: float = Field(default=1.0, gt=0, description="Temperature for smoothing")
    
    # Label configuration
    label_name: str = Field(default='label', description="Name of the label dimension")
    initial_label_list: List[str] = Field(
        default_factory=list,
        description="Initial list of labels (to be added to by the update() routine"
    )
    probability_label_dim_name: str = Field(
        default='probability_label',
        description="Name for probability label dimension"
    )
    
#    # Embedding configuration
#    embedding_dim: Optional[int] = Field(
#        default=None,
#        description="Dimension of embeddings (inferred if None)"
#    )
    
    # Extra label dimensions
    extra_internal_label_dims_list: List = Field(
        default_factory=list,
        description="Additional label dimension values from internal model dimensions"
    )
    extra_internal_label_dims_name_list: List[str] = Field(
        default_factory=list,
        description="Names for additional label dimensions from internal model dimensions"
    )
    
    class Config:
        arbitrary_types_allowed = True



#class Config(BaseModel):
class ModelAnalyzerConfig(BaseModel):
    """Main configuration class containing all sub-configurations."""
    
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    estimator: EntropyEstimatorConfig = Field(default_factory=EntropyEstimatorConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'ModelAnalyzerConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ModelAnalyzerConfig':
        """Load configuration from dictionary."""
        return cls(**config_dict)
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return self.model_dump()





## COMMENTED OUT / UNUSED
_ = '''

## Alias the main ModelAnalyzerConfig class
Config = ModelAnalyzerConfig


class MyModel:
    """Example model class using the configuration."""
    
    def __init__(self, config: Optional[Config] = None, config_path: Optional[str] = None, **kwargs):
        """
        Initialize model with configuration.
        
        Args:
            config: Config object (takes precedence)
            config_path: Path to YAML config file
            **kwargs: Individual parameters to override defaults
        """
        # Priority: config object > config_path > kwargs > defaults
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = Config.from_yaml(config_path)
        elif kwargs:
            self.config = Config.from_dict(kwargs)
        else:
            self.config = Config()
        
        # Now you can access all parameters in a structured way
        self._setup()
    
    def _setup(self):
        """Setup method using the configuration."""
        print("Model initialized with configuration:")
        print(f"  Dataset repo: {self.config.dataset.task_repo}")
        print(f"  Model ID: {self.config.model.model_id}")
        print(f"  Batch size: {self.config.training.batch_size}")
        print(f"  Device: {self.config.training.device}")
        print(f"  Number of bins: {self.config.estimator.n_bins}")
        
        # Your actual setup code here
        # e.g., load model, tokenizer, etc.
    
    def train(self):
        """Example training method."""
        # Access config parameters as needed
        for batch_idx in range(self.config.training.n_batches):
            print(f"Processing batch {batch_idx + 1}/{self.config.training.n_batches}")
            # Your training code here


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("USAGE EXAMPLE 1: Load from YAML file")
    print("=" * 70)
    
    # Method 1: Load from YAML file
    model1 = MyModel(config_path='config.yaml')
    print()
    
    print("=" * 70)
    print("USAGE EXAMPLE 2: Create with default Config")
    print("=" * 70)
    
    # Method 2: Use defaults
    model2 = MyModel()
    print()
    
    print("=" * 70)
    print("USAGE EXAMPLE 3: Override specific parameters")
    print("=" * 70)
    
    # Method 3: Override specific parameters
    config = Config(
        dataset=DatasetConfig(task_repo='custom/repo'),
        training=TrainingConfig(n_batches=20, batch_size=128)
    )
    model3 = MyModel(config=config)
    print()
    
    print("=" * 70)
    print("USAGE EXAMPLE 4: Partial override via dict")
    print("=" * 70)
    
    # Method 4: Partial override via dictionary
    model4 = MyModel(**{
        'training': {'n_batches': 5, 'device': 'cuda:0'},
        'estimator': {'n_bins': 500}
    })
    print()
    
    print("=" * 70)
    print("USAGE EXAMPLE 5: Access and modify configuration")
    print("=" * 70)
    
    # Access configuration
    print(f"Model ID: {model1.config.model.model_id}")
    print(f"Tokenizer max length: {model1.config.tokenizer.max_len}")
    print(f"Estimator smoothing temp: {model1.config.estimator.smoothing_temp}")
    
    # Modify configuration (creates new object, Pydantic models are mutable)
    model1.config.training.n_batches = 15
    print(f"Updated n_batches: {model1.config.training.n_batches}")
    print()
    
    print("=" * 70)
    print("USAGE EXAMPLE 6: Save configuration to new YAML file")
    print("=" * 70)
    
    # Save modified config
    model1.config.to_yaml('config_modified.yaml')
    print("Configuration saved to 'config_modified.yaml'")
    print()
    
    print("=" * 70)
    print("USAGE EXAMPLE 7: Convert to dictionary")
    print("=" * 70)
    
    # Convert to dict
    config_dict = model1.config.to_dict()
    print("Config as dictionary:")
    print(f"  Training seed: {config_dict['training']['seed']}")
    print(f"  Dataset columns: {config_dict['dataset']['data_column_name_list']}")

'''
