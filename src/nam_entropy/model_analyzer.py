
from transformers import BertTokenizerFast, BertModel, AutoTokenizer, AutoModel, CohereModel, CohereConfig
from typing import Optional, List, Literal, Any
from pathlib import Path
import yaml

from datasets import load_dataset, Dataset

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm.notebook import tqdm as nb_tqdm
from tqdm import tqdm
import gc


#from .model_config import Config as ModelAnalyzerConfig
from .model_config import ModelAnalyzerConfig, DatasetConfig, TrainingConfig
from .h2 import EntropyAccumulator2

#from .model_config import DatasetConfig, ModelConfig, TrainingConfig



class ModelAnalyzer(object):
    """
    Class to perform an online entropy analysis of a HuggingFace model.
    """

    def __init__(self, config: ModelAnalyzerConfig = None, config_path: str = None) -> None:
        """
        Initialize the (online) entropy estimation for the model, which updates the internal 
        entropy estimator instance (if passed), or creates a new one (if it's not given).


        ADD DOCUMENTATION HERE!                


        PARAMETER COMMENTS:
            h_estimator -- the online entropy estimator class used to perform entropy estimation -- THIS CAN BE BUILT-IN FOR NOW!
        """                

        # Dataset attributes
#        self.dataset = None
#        self.dataset_name = None
#        self.data_repo = None
#        self.data_subtask = None


        



        # Load config from YAML or use defaults
        if config_path:
            self.config = ModelAnalyzerConfig.from_yaml(config_path)
        elif config:
            self.config = config
        else:
            self.config = ModelAnalyzerConfig()


        ## Specify the Model
        self.model = AutoModel.from_pretrained(self.config.model.model_id, 
                                                cache_dir=f"{self.config.model.cache_dir}/hub", 
                                                output_hidden_states=True,    ## Ensures that the hidden states are passed in the model output
                                                device_map="auto",
                                               ).to(self.config.training.device)

        ## Specify the Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.model_id, 
                                                       cache_dir=f"{self.config.model.cache_dir}/hub",
                                                       model_max_length=self.config.tokenizer.max_len,
                                                       truncation=True, max_length=self.config.tokenizer.max_len,
                                                      )


        ## Specify the DataSet
        self.dataset = self.get_dataset()


        ## Make the DataLoader
        self.dataloader = self.make_dataloader(dataset=self.dataset)


        ## Initialize the Entropy Accumulator
        self.entropy_accumulator = EntropyAccumulator2(config=self.config.entropy_estimator)


        ## Perform the Entropy Analysis
        self.entropy_accumulator = self.analyze_entropies(entropy_accumulator=self.entropy_accumulator, 
                                                          dataloader=self.dataloader)







    def __repr__(self, include_model_id=True):
        ## Define the output string
        #out_str = "ModelAnalyzer class instance"
        out_str = "ModelAnalyzer"
        if include_model_id:
            out_str += f" for the {self.config.model.model_id} model"

        ## Return the desired output string
        return out_str




    _ = """
    def get_dataset(self, repo:str="nyu-mll/glue", subtask:str="mnli"):
        '''
        ADD DOCUMENTATION HERE!                

        
        loads the specified dataset and subtask from hugging face or local cache
        defaults to loading MNLI entailment task from the GLUE benchmark
        
        repo: str - hugging face repo name
        subtask: str (optional) - repo subtask name
        '''
        ## Alias the configuration 
        #dataset_config = self.config.dataset

        self.dataset_name = repo if subtask is None else repo+'/'+subtask
        if subtask is not None:
            self.dataset = load_dataset(repo, subtask, cache_dir=f"{self.cache_dir}/datasets")
        else:
            self.dataset = load_dataset(repo, cache_dir=f"{self.cache_dir}/datasets")

        self.data_repo = repo
        self.data_subtask = subtask
    """

    _ = '''
    def get_dataset(self, dataset_config: Optional[DatasetConfig] = None):
        """Load dataset using DatasetConfig."""
        config = dataset_config or self.config.dataset
        
        source_type = config.get_source_type()
        
        if source_type == 'huggingface':
            return self._load_from_huggingface(config)
        elif source_type == 'local':
            return self._load_from_local(config)
        elif source_type == 'url':
            return self._load_from_url(config)
    '''





    def get_dataset(self, dataset_config: Optional[DatasetConfig] = None):
        """
        Load dataset from HuggingFace, local file, or remote URL.
        
        Args:
            dataset_config: DatasetConfig instance. If None, uses self.config.dataset
            
        Returns:
            datasets.Dataset object
            
        Examples:
            # Use config default
            model.get_dataset()
            
            # Override with new config
            model.get_dataset(DatasetConfig(task_repo='imdb'))
            
            # Load from local file
            model.get_dataset(DatasetConfig(
                task_repo=None,
                data_path='./data/train.csv',
                file_format='csv'
            ))
        """
        # Use provided config or fall back to instance config
        config = dataset_config or self.config.dataset
        
        # Route to appropriate loader based on source type
        source_type = config.get_source_type()
        
        if source_type == 'huggingface':
            return self._load_from_huggingface(config)
        elif source_type == 'local':
            return self._load_from_local(config)
        elif source_type == 'url':
            return self._load_from_url(config)
    



    def _load_from_huggingface(self, config: DatasetConfig):
        """Load dataset from HuggingFace Hub."""
        repo = config.task_repo
        subtask = config.subtask_name
        
        self.dataset_name = repo if subtask is None else f"{repo}/{subtask}"
        self.data_repo = repo
        self.data_subtask = subtask
        
        cache_path = Path(self.config.model.cache_dir) / "datasets"
        cache_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading HuggingFace dataset: {self.dataset_name}")
        
        if subtask is not None:
            self.dataset = load_dataset(repo, subtask, cache_dir=str(cache_path))
        else:
            self.dataset = load_dataset(repo, cache_dir=str(cache_path))
        
        print(f"✓ Loaded with splits: {list(self.dataset.keys())}")
        return self.dataset


    def _load_from_local(self, config: DatasetConfig):
        """Load dataset from local file."""
        file_path = config.data_path
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.dataset_name = path.stem
        self.data_repo = None
        self.data_subtask = None
        
        # Detect file format
        file_format = config.file_format or path.suffix.lstrip('.')
        
        print(f"Loading local file: {file_path} (format: {file_format})")
        
        # Load based on format
        if file_format == 'csv':
            self.dataset = load_dataset('csv', data_files=str(path))
        elif file_format in ['json', 'jsonl']:
            self.dataset = load_dataset('json', data_files=str(path))
        elif file_format == 'parquet':
            self.dataset = load_dataset('parquet', data_files=str(path))
        elif file_format == 'arrow':
            self.dataset = Dataset.from_file(str(path))
        elif file_format == 'txt':
            with open(path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            self.dataset = Dataset.from_dict({'text': lines})
        else:
            raise ValueError(f"Unsupported format: {file_format}")
        
        print(f"✓ Loaded dataset")
        return self.dataset
    
    def _load_from_url(self, config: DatasetConfig):
        """Load dataset from remote URL."""
        url = config.data_url
        
        self.dataset_name = Path(url).stem
        self.data_repo = None
        self.data_subtask = None
        
        # Detect file format
        file_format = config.file_format or Path(url).suffix.lstrip('.')
        
        print(f"Loading remote file: {url} (format: {file_format})")
        
        # Load based on format
        if file_format == 'csv':
            self.dataset = load_dataset('csv', data_files=url)
        elif file_format in ['json', 'jsonl']:
            self.dataset = load_dataset('json', data_files=url)
        elif file_format == 'parquet':
            self.dataset = load_dataset('parquet', data_files=url)
        else:
            raise ValueError(f"Unsupported URL format: {file_format}")
        
        print(f"✓ Loaded dataset")
        return self.dataset
    
    def get_split(self, split_name: Optional[str] = None):
        """Get specific split from loaded dataset."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call get_dataset() first.")
        
        split_name = split_name or self.config.dataset.data_split_name
        
        if split_name not in self.dataset:
            raise ValueError(
                f"Split '{split_name}' not found. "
                f"Available: {list(self.dataset.keys())}"
            )
        
        return self.dataset[split_name]




    ## Helper function to get the batches -- and enhance it with the 'tokenized' and 'tokens' information!
#    def get_batch(self, batch, label_str_to_int_mapping_dict=label_str_to_int_mapping_dict):
    def get_batch(self, batch):

        ## Get the sentence and language information from the dataset
        #words, labels = get_example(batch)
        sentences = [i['sentence'] for i in batch]
        sentence_labels = [i['language'] for i in batch]

        ## Create a batch-specific mapping of label values to 0, 1, ... 
        batch_unique_label_list = list(set(sentence_labels))
        batch_label_str_to_int_mapping_dict = {i: label_i  for label_i, i in enumerate(batch_unique_label_list) }

        ## Create the index tensor for this batch-specific label-index mapping
        sentence_label_index_tensor = torch.tensor([batch_label_str_to_int_mapping_dict[i['language']]  for i in batch])


        ## Create the tokenized version of each sentence
        tokenized = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=256).to(self.config.training.device)

        ## Extract the non-padded token IDs from each tokenized sentence
        tokens = [tokenized['input_ids'][i][tokenized['input_ids'][i].nonzero()].T[0].tolist() 
                    for i in range(len(tokenized['input_ids']))]

        ## Create token-level labels for each token in the sentence (by repeating the sentence-level label)
        if sentence_labels is not None:
            lens = tokenized.attention_mask.sum(-1).tolist()  ## Token lengths needed to spread out the sentence-level labels
            token_labels_list = [x*lens[i] for i, x in enumerate(sentence_labels)]  ## Create the sentence-level labels

        ## Create the label-index tensor associated with the batch:  shape = shape of tokenized['input_ids']
        token_label_index_tensor = sentence_label_index_tensor.unsqueeze(1)
        masked_token_label_index_tensor = torch.where(tokenized.attention_mask == 0, -1, token_label_index_tensor)
        
        ## RETURN KEY:
        _ = '''
        sentence = sentence (as a string -- i.e. list of characters)
        tokenized = transformers.tokenization_utils_base.BatchEncoding object 
                    (i.e. two Pytorch tensors as a dict -- one for tokens and one for attention masks)
        tokens = list of tokens (not padded)
        token_labels = list of token-level labels
        '''    
        
        ## Return the desired words, (padded / masked) tokenizations, (unpadded) token list, (unpadded) token-level labels, and (padded/masked) token_label_index_tensor
        return sentences, tokenized, tokens, token_labels_list, masked_token_label_index_tensor, batch_unique_label_list



    _ = '''
#    def dataloader(self, data_source, batch_size=32, dataloader_kwargs={'shuffle': True}):
    def make_dataloader(self, dataset=None, training_config: Optional[TrainingConfig] = None) -> DataLoader:                        
        """
        Create a DataLoader using the given configuration and self.get_batch().
        
        Args:
            dataset: Dataset to load. If None, uses self.dataset
            training_config: Training configuration. If None, uses self.config.training
            
        Returns:
            DataLoader instance
        """
        # Use provided config or fall back to instance config
        config = training_config or self.config.training
        dataset = dataset or self.dataset
        
        ## Create the dataloader
        my_dataloader = DataLoader(dataset, 
                                   batch_size=config.batch_size,
                                   shuffle=config.shuffle,
                                   num_workers=config.num_workers,
                                   pin_memory=config.pin_memory,
                                   drop_last=config.drop_last,
                                   collate_fn=self.get_batch
                                 )                                   
                                   
        ## Return the dataloader
        return my_dataloader    
    '''


    def make_dataloader(self, 
                        dataset=None, 
                        split_name: Optional[str] = None,
                        training_config: Optional[TrainingConfig] = None) -> DataLoader:
        """
        Create a DataLoader using the given configuration and self.get_batch().
        
        Args:
            dataset: Dataset to load. If None, uses self.dataset
            training_config: Training configuration. If None, uses self.config.training
            
        Returns:
            DataLoader instance
        """
        # Use provided config or fall back to instance config
        config = training_config or self.config.training
        dataset = dataset or self.dataset
        
        ## SANITY CHECK: Is there a dataset present?
        if dataset is None:
            raise ValueError("No dataset provided")
        
        # Check if dataset is a DatasetDict (has splits)
        if hasattr(dataset, 'keys'):
            # It's a DatasetDict - need to select a split
            split_name = split_name or self.config.dataset.data_split_name
            
            if split_name not in dataset:
                available = list(dataset.keys())
                raise ValueError(
                    f"Split '{split_name}' not found. Available splits: {available}"
                )
            
            dataset = dataset[split_name]
            print(f"✓ Using split: '{split_name}'")

        ## Create the dataloader -- only passing non-None parameters!
        return DataLoader(
            dataset,
            collate_fn=self.get_batch,
            **config.dataloader_params
        )




    def process_batch(self, batch):
        """
        Extract and reshape hidden states from a batch.
        """
        words, tokenized, tokens, labels, token_label_tensor, batch_specific_label_list = batch
        
        # Get model output
        output = self.model(**tokenized)
        
        # Stack hidden states
        all_hidden = torch.stack(output.hidden_states, dim=-1)
        
        # Reshape to your format
        data_tensor = all_hidden.flatten(0, 1).transpose(1, -1)
        index_tensor = token_label_tensor.flatten(0, 1)
        
        return data_tensor, index_tensor, batch_specific_label_list
    



    def analyze_entropies(self, entropy_accumulator=None, dataloader=None):
        """
        Analyze various entropies for the given dataset / dataloader
        """
        ## Use the internal dataloader if a dataloader is not passed in.
        if dataloader is None:
            dataloader = self.dataloader

        ## Use the internal entropy_accumulator if a dataloader is not passed in.
        if entropy_accumulator is None:
            entropy_accumulator = self.entropy_accumulator


        ## Loop through all batches to perform the analysis
        for tmp_batch_num, batch in enumerate(dataloader):
            data_tensor, index_tensor, batch_label_list = self.process_batch(batch)

            ## Process each batch of data
            entropy_accumulator.update(data_tensor=data_tensor, batch_index_tensor=index_tensor, batch_label_list=batch_label_list)

            ## REPORTING
            print(f"Processed batch #{tmp_batch_num + 1}")


        ## REPORTING -- Analysis Completed 
        if tmp_batch_num == 0:
            print(f"Finished Entropy Analysis after {tmp_batch_num + 1} batch.")
        else:
            print(f"Finished Entropy Analysis after {tmp_batch_num + 1} batches.")


        ## Return the updated entropy accumulator
        return entropy_accumulator



 






    _ = '''

    def get_dataloader(self, split="train", bs=512, repo="nyu-mll/glue", subtask="mnli", pos_labels=False):

        """
        ADD DOCUMENTATION HERE!                        
        """
        if self.dataset is None:
            self.get_dataset(repo, subtask)
        
        self.bs = bs
        split_loader = DataLoader(
            self.dataset[split], 
            batch_size=bs, 
            shuffle=True, 
            collate_fn=partial(self.get_batch, pos_labels=pos_labels)
        
        )
        return split_loader

    '''


    _ = '''
    def online_estimation(self, 
                          dataloader, 
                          max_batches=None, 
                          additional_logs={}, 
                          label_types=['language','token','bigram'],
                          shift_token=False,
                          start_layer=-2,
                          end_layer=-1,
                         ):
        """
        ADD DOCUMENTATION HERE!                
        """
        #dataloader = self.accelerator.prepare(dataloader)
        if self.is_notebook:
            iterator = nb_tqdm(dataloader, total=max_batches, desc=f"Encoding | {self.dataset_name}")
        else:
            iterator = tqdm(dataloader, total=max_batches, desc=f"Encoding | {self.dataset_name}")

        for ib, data_batch in enumerate(iterator):
            self.batches = ib
            if (max_batches is not None) and (ib >= max_batches):
                break
            #print(self.model)
            #for i in self.model.named_parameters():
            #    print(f"{i[0]} -> {i[1].device}")
            encodings = self.forward(data_batch, start_layer, end_layer)
            if self.h == None:
                subspace_heads = int(self.hidden_size/self.subspace_dim)*self.n_layers
                n_heads = [self.n_layers, int(subspace_heads)]
                print(f"Hidden: {self.hidden_size} | Layers {self.n_layers} | Subspace Heads: {subspace_heads}")
                self.h = self.build_estimators(
                    self.estimator_configs,
                    n_heads=n_heads,
                    heads_per_layer=[int(x/self.n_layers) for x in n_heads],
                    label_classes=label_types,

                )
            if False:
                subprocess.run([
                    "nvidia-smi", 
                ])
            
            all_representations, all_tokens, all_labels = self.get_all_token_representations(encodings)
            label_ids = get_token_encodings(
                examples=all_tokens, 
                labels=label_types, 
                pad_idx=self.pad_idx,
                seq_labels=all_labels, 
            )

            
            for i_estimator in self.h:
                i_estimator.batch_count(all_representations, label_ids)

            if (ib%int(1000/self.bs) == 0):
                
                self.analyse_and_log(additional_logs, ib)
            
            del all_representations
            del encodings
            torch.cuda.empty_cache()
            gc.collect()

        self.analyse_and_log(additional_logs, ib)
        self.summary(additional_logs, ib)
        self.export_categoricals(ib)
    '''


    _ = '''
    def get_batch(self, batch, pos_labels=False):
        """
        This is used in creating the batches for the dataloader to return.

        This means that the dataset is something we create/import separately, 
        but the dataloader is something internal to the model that pulls exactly 
        the data we're interested
        """
        words, labels = self.get_example(batch)

        tokenized = self.tokenizer(
            words, return_tensors='pt', padding=True, 
            truncation=True, max_length=500
        ).to(self.device)

        tokens = [
            tokenized['input_ids'][i][tokenized['input_ids'][i].nonzero()].T[0].tolist() 
                    for i in range(len(tokenized['input_ids']))
        ]

        return words, tokenized, tokens, labels
    '''














## ===============================================================
## ===============================================================
## ===============================================================

_ = """

import torch
from torch.utils.data.dataloader import DataLoader

import attr
import gc
import json
import os
#import spacy
import bisect
import subprocess
import logging

import wandb

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
from datetime import date as D
from itertools import tee
from collections import defaultdict
from functools import partial
from os import makedirs


from datasets import load_dataset
from tqdm.notebook import tqdm as nb_tqdm
from tqdm import tqdm
from huggingface_hub import login

from accelerate import Accelerator, load_checkpoint_and_dispatch, init_empty_weights

from transformers import BertTokenizerFast, BertModel, AutoTokenizer, AutoModel, CohereModel, CohereConfig
from entropy.estimation import Information

@attr.s
class EncoderState:
    '''
    An an object for storing the output of An Encoder model
    '''
    src_memory = attr.ib()
    tokens = attr.ib()
    mask = attr.ib()
    labels = attr.ib(default=[None])
    pos_labels=attr.ib(default=[None])
    dep_labels=attr.ib(default=[None])
        
class Analyser(object):
    def __init__(
        self, device:str ='cpu', h_estimator:Information=None, model_id:str='google/multiberts-seed_13',
        results_path:str='results', cache_dir:str=None, max_len:int=256,
        seed:int=999999, is_jupyter_nb:bool=False, spacy_model='en_core_web_sm', use_cls=False,
        estimator_configs:list=[],subspace_dim=64, data_path=None
    ) -> None:
        '''
        Class for Loading and Analysing a BERT Model
        
        This init function sets object variables, builds a defailt estimator if none is provided
        Sets global plot aesthetics, Sets torch random seed, Then attempts to load the specified
        model using HuggingFace Transformers
        
        device: str - pytorch device to load model into. usually cpu or cuda:0
        
        h_estimator: Estimator - a class imported from the h_estimator file in this directory
        it's a class for fast pytorch-based soft entropy estimation
        
        model_id: str - hugging face model path to load (needs to be a BERTmodel, an error is thrown otherwise)
        
        results_file_name:str - the relative path where a json file with any results will be automatically written
        
        cache_dir: str - path to where hugging face should cache resources
        
        max_len: int - maximum length the tokenizer will allow, any longer sequences this will be truncated
        
        seed: int - sets pytorch random seed for controlling randomness in data shuffle and estimation
        
        is_jupyter_nb : bool - renders progress bars in a notebook-friendly widget
    
        '''
        
        torch.cuda.empty_cache()
        gc.collect()

        self.accelerator = Accelerator()
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #self.device = self.accelerator.device
        
        self.dataset=None
        self.cache_dir=cache_dir

        model_name = os.path.basename(os.path.normpath(model_id))
        results_path = os.path.join(results_path,model_name)
        
        wandb.init(
            job_type='analysis',
            project='language_id_flores',
            group=model_name,
            entity='cohere',
            config={
                'model':model_id,
                'data':data_path,
                'estimators':estimator_configs,
                'results_path': results_path

            }
        )

        date = D.today()
        self.save_path = f"{results_path}/results.json"
        self.save_dir = f"{results_path}/export"

        makedirs(results_path, exist_ok=True)
        makedirs(self.save_dir, exist_ok=True)
        makedirs(self.cache_dir, exist_ok=True)
        
        self.model_name = model_id
        self.is_notebook = is_jupyter_nb
        self.use_cls = use_cls
        
        #if spacy_model is not None:
        #    self.spacy_tagger = spacy.load(spacy_model)
        self.estimator_configs = estimator_configs
        self.subspace_dim = subspace_dim

        self.h = h_estimator
            
        sns.set(style='whitegrid', font='Arial')
        
        torch.manual_seed(seed)


        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            cache_dir=f"{self.cache_dir}",
            #model_max_length=max_len,
            #truncation=True, max_length=max_len,
            
        )

        
        self.model = AutoModel.from_pretrained(
            model_id, 
            cache_dir=f"{self.cache_dir}", 
            output_hidden_states=True,
            device_map="auto",
        )
        
        if self.tokenizer.pad_token:
            self.pad_idx = self.tokenizer.get_added_vocab()[self.tokenizer.pad_token]
        

        else:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.pad_idx = self.tokenizer.get_added_vocab()[self.tokenizer.pad_token]

    
        #self.model = self.model.to('cuda')

    def build_estimators(
            self, estimator_versions, n_heads=[], 
            heads_per_layer=[], label_classes=[]
        ):
        

        hh = []
        for v in estimator_versions:
            for i, head in enumerate(n_heads):
                hh.append(
                    Information(
                        n_bins=v['n_bins'], 
                        count_device='cpu',
                        n_heads=head, heads_per_layer=heads_per_layer[i],
                        temp=v['temp'], dist_fn=v['distance_fn'],
                        smoothing_fn=v['smoothing_fn'], bin_type=v['bin_type'],
                        label_classes=label_classes,
                    )
                )

        return hh
    
            
    def get_dataset(self, repo:str="nyu-mll/glue", subtask:str="mnli"):
        '''
        loads the specified dataset and subtask from hugging face or local cache
        defaults to loading MNLI entailment task from the GLUE benchmark
        
        repo: str - hugging face repo name
        subtask: str (optional) - repo subtask name
        '''
        self.dataset_name = repo if subtask is None else repo+'/'+subtask
        if subtask is not None:
            self.dataset = load_dataset(repo, subtask, cache_dir=f"{self.cache_dir}/datasets")
        else:
            self.dataset = load_dataset(repo, cache_dir=f"{self.cache_dir}/datasets")

        self.data_repo = repo
        self.data_subtask = subtask




    def get_example(self, batch):
        '''
        This method is called every batch and formats the data for ech specific dataset into
        the batch format needed by the get batch method. It returns a list of lists with each
        example sentence or pair
        '''
        
        if self.data_subtask in ['mnli', 'ax']:
            #return [i['premise'] for i in batch]
            return [' '.join([i['premise'], i['hypothesis']]) for i in batch], None
        
        if self.data_subtask in ['rte', 'mrpc', 'stsb', 'wnli']:
            return [' '.join([i['sentence1'], i['sentence2']]) for i in batch], None
        
        if self.data_subtask in ['qnli', 'sst2']:
            return [' '.join([i['question'], i['sentence']]) for i in batch], None
        
        if self.data_subtask in ['qqp']:
            return [' '.join([i['question1'], i['question2']]) for i in batch], None
        
        if self.data_subtask in ['cola', 'sst2']:
            return [i['sentence'] for i in batch], None
        
        if self.data_repo =="sentence-transformers/wikipedia-en-sentences":
            return [i['sentence'] for i in batch], None
         
        if 'mix' in self.data_repo:
            return [i['sentence'] for i in batch], [[i['language']] for i in batch]
        
        if 'flores' in self.data_repo:
            return [i['sentence'] for i in batch], [i['language_id'] for i in batch]
        
        if 'hcoxec' in self.data_repo:
            return [i['sentence'] for i in batch], None
        
        raise NotImplementedError
        
    def get_batch(self, batch, pos_labels=False):
        words, labels = self.get_example(batch)

        tokenized = self.tokenizer(
            words, return_tensors='pt', padding=True, 
            truncation=True, max_length=500
        ).to(self.device)

        tokens = [
            tokenized['input_ids'][i][tokenized['input_ids'][i].nonzero()].T[0].tolist() 
                    for i in range(len(tokenized['input_ids']))
        ]

        return words, tokenized, tokens, labels

        
        
    def get_dataloader(self, split="train", bs=512, repo="nyu-mll/glue", subtask="mnli", pos_labels=False):
        if self.dataset is None:
            self.get_dataset(repo, subtask)
        
        self.bs = bs
        split_loader = DataLoader(
            self.dataset[split], 
            batch_size=bs, 
            shuffle=True, 
            collate_fn=partial(self.get_batch, pos_labels=pos_labels)
        
        )
        return split_loader
    
    def get_all_token_representations(self, encodings):
        
        all_representations = encodings.src_memory[encodings.mask.gt(0)]
        return all_representations, encodings.tokens, encodings.labels


            
    def forward(self, x, start_layer, end_layer):
        
        words, tokenized, tokens, labels = x
        with torch.no_grad():
            tokenized = tokenized.to(self.device)
            output = self.model(**tokenized)
            all_hidden = torch.stack(output.hidden_states[start_layer:end_layer], dim=-1)
            hidden_shape = all_hidden.shape
            if self.batches == 0:
                self.hidden_size = hidden_shape[-2]
                self.n_layers = hidden_shape[-1]
            #logging.info(f"hidden shape: {hidden_shape}")
            collapsed_column = all_hidden.view(
                hidden_shape[0],
                hidden_shape[1], 
                -1
            )
            enc_state = EncoderState(
                src_memory=collapsed_column,
                tokens=tokens,
                mask=tokenized['attention_mask'],
                labels=labels,
            )
            self.enc_state = enc_state
        return enc_state

    
    def online_estimation(
        self, 
        dataloader, 
        max_batches=None, 
        additional_logs={}, 
        label_types=['language','token','bigram'],
        shift_token=False,
        start_layer=-2,
        end_layer=-1,
        
        
    ):
        
        #dataloader = self.accelerator.prepare(dataloader)
        if self.is_notebook:
            iterator = nb_tqdm(dataloader, total=max_batches, desc=f"Encoding | {self.dataset_name}")
        else:
            iterator = tqdm(dataloader, total=max_batches, desc=f"Encoding | {self.dataset_name}")

        for ib, data_batch in enumerate(iterator):
            self.batches = ib
            if (max_batches is not None) and (ib >= max_batches):
                break
            #print(self.model)
            #for i in self.model.named_parameters():
            #    print(f"{i[0]} -> {i[1].device}")
            encodings = self.forward(data_batch, start_layer, end_layer)
            if self.h == None:
                subspace_heads = int(self.hidden_size/self.subspace_dim)*self.n_layers
                n_heads = [self.n_layers, int(subspace_heads)]
                print(f"Hidden: {self.hidden_size} | Layers {self.n_layers} | Subspace Heads: {subspace_heads}")
                self.h = self.build_estimators(
                    self.estimator_configs,
                    n_heads=n_heads,
                    heads_per_layer=[int(x/self.n_layers) for x in n_heads],
                    label_classes=label_types,

                )
            if False:
                subprocess.run([
                    "nvidia-smi", 
                ])
            
            all_representations, all_tokens, all_labels = self.get_all_token_representations(encodings)
            label_ids = get_token_encodings(
                examples=all_tokens, 
                labels=label_types, 
                pad_idx=self.pad_idx,
                seq_labels=all_labels, 
            )

            
            for i_estimator in self.h:
                i_estimator.batch_count(all_representations, label_ids)

            if (ib%int(1000/self.bs) == 0):
                
                self.analyse_and_log(additional_logs, ib)
            
            del all_representations
            del encodings
            torch.cuda.empty_cache()
            gc.collect()

        self.analyse_and_log(additional_logs, ib)
        self.summary(additional_logs, ib)
        self.export_categoricals(ib)
            

    def summary(self, additional_logs, n_batches, export=False):
        for_wandb = {}
        for i, i_estimator in enumerate(self.h):
            to_log = {}
            to_log['model_id'] = self.model_name
            to_log['dataset'] = self.dataset_name
            to_log['n_examples'] = n_batches*self.bs
            to_log['n_batches'] = n_batches
            to_log['n_bins'] = i_estimator.n_bins
            to_log['n_heads'] = i_estimator.n_heads
            to_log['temp'] = i_estimator.temp
            to_log['heads_per_layer'] = i_estimator.heads_per_layer
            to_log['hidden_size'] = self.hidden_size
            to_log['model_layers'] = self.n_layers

            analysis = i_estimator.analyse()
            wandb_name = 'subspace' if i_estimator.heads_per_layer ==1 else 'layer'
            wandb_name = f"{wandb_name}_{i_estimator.n_bins}"

            for result in analysis:
                wandb.run.summary[f"{wandb_name}/{result}"] = analysis[result]

            

        wandb.log(for_wandb, step=n_batches*self.bs)

    def analyse_and_log(self, additional_logs, n_batches, export=False):
        for_wandb = {}
        for i, i_estimator in enumerate(self.h):
            to_log = {}
            to_log['model_id'] = self.model_name
            to_log['dataset'] = self.dataset_name
            to_log['n_examples'] = n_batches*self.bs
            to_log['n_batches'] = n_batches
            to_log['n_bins'] = i_estimator.n_bins
            to_log['n_heads'] = i_estimator.n_heads
            to_log['temp'] = i_estimator.temp
            to_log['heads_per_layer'] = i_estimator.heads_per_layer
            to_log['hidden_size'] = self.hidden_size
            to_log['model_layers'] = self.n_layers

            analysis = i_estimator.analyse()
            wandb_name = 'subspace' if i_estimator.heads_per_layer ==1 else 'layer'
            wandb_name = f"{wandb_name}_{i_estimator.n_bins}"

            for result in analysis:
                for_wandb[f"{wandb_name}/{result}"] = analysis[result]

            self.record(analysis, additional=to_log)
            

        wandb.log(for_wandb, step=n_batches*self.bs)
    
    def filter_results(
        self, results:dict, 
        filters=['regularity','variation','disentanglement', 'residual']
    ) -> dict:
        '''
        returns only results whose keys in the results dict contain at least
        one of the filter strings. This is done for readability, in the event that
        the number of results from the analysis is too large
        '''
        to_return = {}
        for item in results:
            for filt in filters:
                if filt in item:
                    to_return[item] = results[item]
                    break
        
        return to_return
    
    def build_df(self, results:dict) -> pd.DataFrame:
        '''
        Constructs a Pandas DataFrame object from a dictionary of results
        Assumes that the results dictionary keys a strings with layout
        'measure/label' where measure is the 
        '''
        items = defaultdict(lambda: defaultdict(lambda: []))
        labels = []
        for i in results:
            measure, label = i.split('/')
            items[measure][label] = results[i]
            #items['label'][label] = label
            labels.append(label)
        df = pd.DataFrame.from_dict(items)
        df['label'] = df.index.values
        
        return df
    
    def record(self, data: dict, additional: dict = {}) -> None:
        '''
        Writes data to outfile, after adding any additional keys
        passed in from the additional dict. These can be parameters from
        the analysis - like number of samples, or a readable name of the model
        
        Note: the model id and dataset saved in the analyser object are 
        automatically added
        
        '''
        additional['model_id'] = self.model_name
        additional['dataset'] = self.dataset_name

        additional.update(data)

        with open(self.save_path, "a") as outfile:
            outfile.write(json.dumps(additional)+'\n')

    def export_categoricals(self, n_batches:int, additional: dict = {}) -> None:
        '''
        Exports categorical distributions returned by the estimator
        '''
        n_examples = n_batches*self.bs
        for i, i_estimator in enumerate(self.h):
            
            additional = {}
            additional['model_id'] = self.model_name
            additional['dataset'] = self.dataset_name
            additional['n_examples'] = n_examples
            additional['n_bins'] = i_estimator.n_bins
            additional['n_heads'] = i_estimator.n_heads
            additional['temp'] = i_estimator.temp
            additional['heads_per_layer'] = i_estimator.heads_per_layer
            additional['hidden_size'] = self.hidden_size
            additional['model_layers'] = self.n_layers

            categoricals = i_estimator.export()

            overall = {
                'counts': categoricals['overall'],
                'config':additional,
            }

            self.dump(f'overall_estimator_{i}_ex_{n_examples}.json', overall)

            for label in (pbar:= tqdm(categoricals['conditionals'], desc=f"Estimator {i} | writing conditionals")):
                pbar.set_postfix_str(label)
                overall = {
                    'counts': categoricals['conditionals'][label],
                    'config': additional,
                }
                self.dump(f'{label}_estimator_{i}_ex_{n_examples}.json', overall)


                '''ith open(
                    os.path.join(self.save_dir, f'{label}_estimator_{i}_ex_{n_examples}.json'), 
                    "a"
                ) as outfile:
                    outfile.write(json.dumps(additional, indent=4))'''

    def dump(self, path, data):
        with open(
            os.path.join(self.save_dir, path), 
            "w"
        ) as outfile:
            outfile.write(json.dumps(data, indent=4))
            
    def get_time(self):
        return datetime.now().strftime("%H_%M_%S")
            


    

                
                    
    
def n_gram(sequence, n):
    iterables = tee(sequence, n)

    for i, sub_iterable in enumerate(iterables):  # For each window,
        for _ in range(i):  # iterate through every order of ngrams
            next(sub_iterable, None)  # generate the ngrams within the window.
            
    return list(zip(*iterables))  # Unpack and flattens the iterables.

def idx_ngrams(max_len, n):
    idxs = n_gram(range(max_len), n=n)
    overlap_matrix = []
    for i in range(max_len):
        ii = []
        for inn, ngr in enumerate(idxs):
            if i in ngr:
                ii.append(inn)
        overlap_matrix.append(ii)
        
    return overlap_matrix

def get_token_encodings(
    examples, 
    labels=['token','bigram','trigram'],
    pos_dict=None,
    seq_labels=None,
    pad_idx=1,
    parallel=False,
):
    token_ids = defaultdict(lambda: [])
    pos_ids = defaultdict(lambda: [])
    #bigram_ids = defaultdict(lambda: defaultdict(lambda: []))
    #trigram_ids = defaultdict(lambda: defaultdict(lambda: []))
    bigram_ids = defaultdict(lambda: [])
    trigram_ids = defaultdict(lambda: [])
    bow_ids = defaultdict(lambda: defaultdict(lambda: []))
    language_ids = defaultdict(lambda: [])
    general_label_ids = defaultdict(lambda: [])
    
   
    
    i = 0 #assign an id to each token instance in the dataset
    for i_e, example in enumerate(examples):
        tokens = example
        if 'bigram' in labels:
             bigrams, bigram_idxs = n_gram(tokens, n=2), idx_ngrams(len(tokens), n=2)

        if 'trigram' in labels: 
            trigrams, trigram_idxs = n_gram(tokens, n=3), idx_ngrams(len(tokens), n=3)
            
        for i_t, token in enumerate(tokens):
            if token == pad_idx: continue
            #token = token.lower() if uncased else token
            token_ids[token].append(i)
            
            if ('pos' in labels) and (pos_dict is not None):
                token_pos = pos_dict[token]
                pos_ids[token_pos].append(i)

            if ('language' in labels) and (seq_labels is not None):
                language_ids[seq_labels[i_e]].append(i)
                
            if ('label' in labels) and (seq_labels is not None):
                general_label_ids[seq_labels[i_e][i_t]].append(i)
            
                
            if 'bigram' in labels:
                for bi in bigram_idxs[i_t]:
                    bigram_ids[f"{token}_{bigrams[bi]}"].append(i)
            
            if 'trigram' in labels:
                for tri in trigram_idxs[i_t]:
                    trigram_ids[f"{token}_{trigrams[tri]}"].append(i)
            
            if 'bow' in labels:
                for i_tt, other_token in enumerate(tokens):
                    if i_t != i_tt:
                        bow_ids[token][other_token].append(i)
                            
            i+=1
            
    ids = {
        'token': token_ids,
        'pos':pos_ids,
        'bigram':bigram_ids,
        'trigram':trigram_ids,
        'language':language_ids,
        'bow':bow_ids,
    }
    
    label_ids = {}
    for id_type in labels:
        if parallel:
            label_ids[id_type] = list(ids[id_type].items())
        else:
            label_ids[id_type] = ids[id_type]
    
    return label_ids

"""