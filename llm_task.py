import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import torch.nn.functional
import json
from tqdm import tqdm
import os.path
import errno
import matplotlib.pyplot as plt
import os
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# def MI(answer_prob):
#     uniform_prob = torch.ones_like(answer_prob) / len(answer_prob)
#     mi = torch.sum(answer_prob * torch.log2(torch.div(answer_prob, uniform_prob.to(answer_prob.device))))
#     return mi.to('cpu')


class llm_contextual_bandit:
    def __init__(self, dataset, n_arms=2):
        self.n_arms = n_arms
        self.dataset = dataset
        self.n_contexts = len(dataset)
        self.context = None
        self.ind = 0

    def reset(self):
        self.ind = 0
        self.context = None

    def draw_context(self):
        x = {
            'question': self.dataset['question'][self.ind],
            'choices': self.dataset['choices'][self.ind],
            'answer': self.dataset['answer'][self.ind]
        }
        self.ind += 1
        self.context = x
        return x

    def play(self, prob_vector):
        real_answer = self.context['answer']
        reward = prob_vector[real_answer]
        return reward, real_answer


class llm_selecting_agent:
    def __init__(self, tokenizer, model, type='mistral'):
        self.tokenizer = tokenizer
        self.model = model
        self.accum_prompt = ""
        if type == 'mistral':
            self.question_prompt = (
                "You will answer the following question using one of the following letters, A, B, C, or D."
                "Do not explain or describe the answer."
                "You are given the following question:\n")
            self.answers_prompt = "The possible answers are:\n"
            self.answer_labels = ['A', 'B', 'C', 'D']
            self.end_prompt = ("Please output only the letter corresponding with the correct answer - A, B, C or D. "
                               "Don't explain or describe the answer."
                               "\nYour answer:\n")
        elif type == 'llama':
            self.question_prompt = (
                "You are a bot that only outputs one of the following letters - A, B, C or D."
                "You are designed to answer multiple choice questions."
                "Do not explain or describe the answer.\n"
                "Question: You are given the following question:\n"
            )
            self.answers_prompt = "The possible answers are:\n"
            self.answer_labels = ['A', 'B', 'C', 'D']
            self.end_prompt = ("You must output only the letter corresponding with the correct answer - A, B, C or D. "
                               "Don't explain or describe the answer.\n"
                               "Output:\n")
        elif type == 'falcon':
            self.question_prompt = (
                "You are an AI assistant designed to answer multiple choice questions."
                "You will answer the following question using one of the following letters, A, B, C, or D."
                "Do not explain or describe the answer."
                "You are given the following question:\n"
            )
            self.answers_prompt = "The possible answers are:\n"
            self.answer_labels = ['A', 'B', 'C', 'D']
            self.end_prompt = ("Please output only the letter corresponding with the correct answer - A, B, C or D. "
                                 "Don't explain or describe the answer."
                                    "\nYour answer:\n")
        elif type == 'gemma':
            self.question_prompt = (
                "<start_of_turn>user\n"
                "You will answer the following question using one of the following letters, A, B, C, or D."
                "Do not explain or describe the answer."
                "You are given the following question:\n")
            self.answers_prompt = "The possible answers are:\n"
            self.answer_labels = ['A', 'B', 'C', 'D']
            self.end_prompt = ("Please output only the letter corresponding with the correct answer - A, B, C or D. "
                               "Don't explain or describe the answer.<end_of_turn>\n"
                               "<start_of_turn>model\n")
        else:
            self.question_prompt = (
                "You will answer the following question using one of the following letters, A, B, C, or D."
                "Do not explain or describe the answer."
                "You are given the following question:\n")
            self.answers_prompt = "The possible answers are:\n"
            self.answer_labels = ['A', 'B', 'C', 'D']
            self.end_prompt = ("Please output only the letter corresponding with the correct answer - A, B, C or D. "
                               "Don't explain or describe the answer."
                               "\nYour answer:\n")
        self.answer_tokens = self.tokenizer.convert_tokens_to_ids(self.answer_labels)
    def reset(self):
        self.accum_prompt = ""
        torch.cuda.empty_cache()

    def prep_question_prompt(self, context):
        ques_prompt = (self.question_prompt + context['question'] + '\n'
                             + self.answers_prompt + '\n'.join([self.answer_labels[i] + ') ' + context['choices'][i] for i in range(4)])
                                + '\n' + self.end_prompt)
        return ques_prompt

    def get_model_output(self, model_id, input_ids):
        with torch.no_grad():
            # outputs = model_id.generate(input_ids.to('cuda'), max_new_tokens=1, output_scores=True,
            #                          return_dict_in_generate=True, pad_token_id=self.tokenizer.pad_token_id,
            #                          attention_mask=torch.ones_like(input_ids).to('cuda'), use_cache=True)
            outputs = model_id.generate(input_ids, max_new_tokens=1, output_logits=True,
                                        pad_token_id=self.tokenizer.pad_token_id, return_dict_in_generate=True,
                                        attention_mask=torch.ones_like(input_ids).to('cuda'), use_cache=True)
            scores = torch.stack(outputs['logits']).squeeze()
            prob_values = torch.nn.functional.softmax(scores[self.answer_tokens], dim=0)
            torch.cuda.empty_cache()
            return prob_values

    def select_action(self, context):
        self.accum_prompt = self.prep_question_prompt(context)
        input_ids = self.tokenizer(self.accum_prompt, return_tensors='pt')['input_ids'].to('cuda')
        prob_values = self.get_model_output(self.model, input_ids)
        # mi = MI(prob_values)
        return prob_values

    def run_trial(self, bandit, episodes):
        losses = np.zeros((episodes,))
        # mis = np.zeros((episodes,))
        for i in tqdm(range(episodes)):
            torch.cuda.empty_cache()
            context = bandit.draw_context()
            prob_values = self.select_action(context)
            reward, real_answer = bandit.play(prob_values)
            losses[i] = 1 - reward
            # mis[i] = mi
        return losses

    def run_trials(self, bandit, episodes, trials):
        losses = np.zeros((trials, episodes))
        # mis = np.zeros((trials, episodes))
        for i in tqdm(range(trials)):
            bandit.reset()
            losses[i] = self.run_trial(bandit, episodes)
            self.reset()
        return losses


def get_args():
    parser = argparse.ArgumentParser(description='Run LLM Contextual Bandit')
    parser.add_argument('--large', action='store_true', help='Use large model')
    parser.add_argument('--n_trials', type=int, default=1, help='Number of trials to run')
    parser.add_argument('--n_episodes', type=int, default=100, help='Number of episodes to run')
    parser.add_argument('--seed', type=int, default=[667], help='Random seed', nargs='+')
    parser.add_argument('--type', type=str, default='mistral', help='Type of model')
    parser.add_argument('--subject', type=str, default='abstract_algebra', help='Subject to use')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    SEED = args.seed
    seed_list = args.seed
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    small_model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
    small_model_llama = 'meta-llama/Meta-Llama-3-8B-Instruct'
    small_model_gemma = 'google/gemma-1.1-2b-it'
    large_model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    large_model_llama = 'meta-llama/Meta-Llama-3-70B-Instruct'
    large_model_gemma = 'google/gemma-1.1-7b-it'
    small_model_falcon = 'tiiuae/falcon-7b-instruct'
    if args.large:
        if args.type == 'llama':
            model_name = large_model_llama
            model_size = 'large_llama'
        elif args.type == 'gemma':
            model_name = large_model_gemma
            model_size = 'large_gemma'
        else:
            model_name = large_model_name
            model_size = 'large_mistral'
    else:
        if args.type == 'llama':
            model_name = small_model_llama
            model_size = 'small_llama'
        elif args.type == 'gemma':
            model_name = small_model_gemma
            model_size = 'small_gemma'
        elif args.type == 'falcon':
            model_name = small_model_falcon
            model_size = 'small_falcon'
        else:
            model_name = small_model_name
            model_size = 'small_mistral'
    print(model_size, model_name)


    if args.type == 'falcon':
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', quantization_config=nf4_config,
                                                     torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', quantization_config=nf4_config,
                                                     torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2',
                                                     trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = load_dataset('cais/mmlu', 'all', split='test').filter(lambda x: x['subject'] == args.subject)
    args.n_episodes = min(args.n_episodes, len(dataset))
    # Take a random split of the dataset
    for seed in seed_list:
        dataset = dataset.shuffle(seed=seed)
        partial_data = dataset[:args.n_episodes]
        bandit = llm_contextual_bandit(partial_data)
        agent = llm_selecting_agent(tokenizer, model, type=args.type)
        file_name = 'results/llm_task_{ms}_{s}_{h}_{subj}.json'.format(ms=model_size, s=seed, h=args.n_episodes, subj=args.subject)
        losses = agent.run_trials(bandit, args.n_episodes, args.n_trials)
        if os.path.exists(os.path.dirname(file_name)) == False:
            try:
                os.makedirs(os.path.dirname(file_name))
            except OSError as exc: # Guard
                if exc.errno != errno.EEXIST:
                    raise
        with open(file_name, 'w') as f:
            f.write(json.dumps([{'rewards': losses[i].tolist()} for i in range(args.n_trials)]))
        print('Results written to ' + file_name)
