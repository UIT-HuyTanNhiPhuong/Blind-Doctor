import os
import yaml
from dotenv import load_dotenv
import torch

from new_rag.pipeline import create_chain
from new_rag.utils import print_color

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

## PROJECT SETUP
device = 'cuda' if torch.cuda.is_available() else 'cpu'

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACE_API_TOKEN")

if os.path.exists('config.yaml'):
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    print_color(f"[INFO] Successfully loaded config.yaml", 'green')


rag_chain = create_chain(config)
answer = rag_chain.invoke("Tôi bị đau bụng dữ dội, tôi có thể bị bệnh gì?").split("Answer: ")

print_color("Context:", 'blue')
print(answer[0])

print_color("Answer:", 'blue')
print(answer[1])
