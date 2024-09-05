import os
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from llm_task import run_expr
from decouple import config
from huggingface_hub import login

SEED = 42
SAVE_PATH = 'results.csv'

login(token=config('HF_TOKEN'))

set_seed(SEED)

mmlu_subjects = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']

if not os.path.exists(SAVE_PATH):
    with open(SAVE_PATH, 'w') as f:
        f.write(f"hf_model_path, {', '.join(mmlu_subjects)}, avg\n") 

hf_model_paths = [
    #### TODO models
    # "mistralai/Mistral-7B-Instruct-v0.1",
    # "mistralai/Mathstral-7B-v0.1",
    # "EleutherAI/llemma_7b",
    # "mistralai/Mistral-7B-Instruct-v0.1_quantized",

    # "meta-llama/Meta-Llama-3-70B-Instruct",
    # "meta-llama/Meta-Llama-3-70B-Instruct_quantized",
    # "meta-llama/Llama-2-7b-chat-hf",
    # "meta-llama/Llama-2-7b-chat-hf_quantized",
    # "NEU-HAI/Llama-2-7b-alpaca-cleaned",
    # "meta-llama/Meta-Llama-3-8B-Instruct",
    # "meta-llama/Meta-Llama-3-8B-Instruct_quantized",
    
    # "CorticalStack/mistral-7b-alpaca-sft",
    # "dicta-il/dictalm2.0",


    # "microsoft/Phi-3.5-mini-instruct",
    # "microsoft/Phi-3.5-mini-instruct_quantized",
    # "microsoft/Phi-3.5-MoE-instruct",
    # "microsoft/Phi-3.5-MoE-instruct_quantized",

    # "google/gemma-1.1-2b-it",
    # 'google/gemma-1.1-2b-it_quantized',
    # 'google/gemma-1.1-7b-it',
    # 'google/gemma-1.1-7b-it_quantized',   
]

types = [
    "mistral" if "mistral" in hf_model_path.lower() else
    "llama" if "llama" in hf_model_path.lower() else
    "mistral" if "dictalm" in hf_model_path.lower() else
    "phi3" if "phi" in hf_model_path.lower() else
    "gemma" if "gemma" in hf_model_path.lower()
    else "unknown"
    for hf_model_path in hf_model_paths
]


for hf_model_path in tqdm(hf_model_paths):
    print(hf_model_path, end='\t|\t')
    if hf_model_path.endswith('_quantized'):
        tokenizer = AutoTokenizer.from_pretrained(hf_model_path.split("_quantized")[0], use_fast=True)
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(hf_model_path.split("_quantized")[0], device_map='auto', quantization_config=nf4_config,
                                                     torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2',
                                                     trust_remote_code=True)
    
    else:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(hf_model_path, device_map='auto',
                                                     torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2',
                                                     trust_remote_code=True)
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_results = {}
    for mmlu_subject in mmlu_subjects:
        mean_loss, _ = run_expr(tokenizer, model, mmlu_subject, SEED, type=types[hf_model_paths.index(hf_model_path)])
        model_results[mmlu_subject] = mean_loss
    
    avg_loss = sum(model_results.values())/len(model_results)

    with open(SAVE_PATH, 'a') as f:
        f.write(f"{hf_model_path}, {', '.join([str(model_results[subject]) for subject in mmlu_subjects])}, {avg_loss}\n")
