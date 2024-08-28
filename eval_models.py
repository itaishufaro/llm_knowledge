from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from llm_task import run_expr

SEED = 42
SAVE_PATH = 'results.csv'

set_seed(SEED)
with open(SAVE_PATH, 'w') as f:
    f.write('hf_model_path,mmlu_subject,mean_loss,std_loss\n')

hf_model_paths = [
    #"migaraa/Alpaca_Llama-3-8B-Instruct_Gaudi",
    "dicta-il/dictalm2.0",
    "yam-peleg/Hebrew-Mistral-7B",
    #"microsoft/Phi-3.5-MoE-instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "google/gemma-1.1-2b-it",
    "tiiuae/falcon-7b-instruct",
    #"mistralai/Mixtral-8x7B-Instruct-v0.1",
    #'meta-llama/Meta-Llama-3-70B-Instruct',
    'google/gemma-1.1-7b-it',
]

mmlu_subjects = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']

for hf_model_path in tqdm(hf_model_paths):
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, use_fast=True)
    if "falcon" not in hf_model_path:
        model = AutoModelForCausalLM.from_pretrained(hf_model_path, device_map='auto', quantization_config=nf4_config,
                                                     torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2',
                                                     trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(hf_model_path, device_map='auto', quantization_config=nf4_config,
                                                     torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    for mmlu_subject in tqdm(mmlu_subjects, leave=False):
        mean_loss, std_loss = run_expr(tokenizer, model, mmlu_subject, SEED)
        with open(SAVE_PATH, 'a') as f:
            f.write(f'{hf_model_path},{mmlu_subject},{mean_loss},{std_loss}\n')
        break
