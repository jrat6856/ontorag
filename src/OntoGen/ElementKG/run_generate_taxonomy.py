# set random seed for reproducibility
import random
import numpy as np
import torch
import sys, os

seed = 0 # seeds from 0 to 5

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

sys.path.append('../')
from generate_taxonomy import generate_taxonomy

import dotenv
import os

# Reload the variables in your '.env' file (override the existing variables)
dotenv.load_dotenv("/home/oarga/.env", override=True)


txt_files_path = 'docs/plain_text'
txt_files = [os.path.join(txt_files_path, f) for f in os.listdir(txt_files_path) if f.endswith('.txt')]

model = 'llama3.1:70b'
#model = 'meta-llama/llama-3.1-70b-instruct'

options = {
    'temperature': 0.1,
    'num_predict': 4096,
    'seed': seed,
    'num_ctx': 32000
}
'''
options = {
    'temperature': 0.1,
    #'top_k': 40,
    'top_p': 0.9,
    'max_tokens': 32000,
    'seed': seed,
    #'num_ctx': 32000
}
'''
num_iterations = 4

generate_taxonomy(
    root_dir='docs/plain_text',
    category_seed_file='docs/categories.txt',
    txt_files=txt_files,
    model=model,
    model_params=options,
    num_iterations=num_iterations,
    prompt_include_path=True,
    backend='ollama',
    base_url='https://openrouter.ai/api/v1',
    seed=seed,
    sc_retry=1,
    majority=1
)