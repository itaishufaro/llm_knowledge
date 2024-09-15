# Comparing LLM Knowledge With H-Divergence
*Project for 048100 - Reliability in Modern Machine Learning*

## Running Experiments
### MMLU Evaluation
In the file ```eval_models.py```, Specify HuggingFace model paths in ```hf_model_paths```.


Execute ```python ./eval_models.py```.

Make sure you have enough GPU memory for the required models.


### H-Divergence Report
To get a CSV report of the average H-Divergence and Top/Bottom-5 subjects:

In the file ```compare_models.py```, specify the compared models, e.g.:
```
comparisons = [
    ("mistralai/Mistral-7B-Instruct-v0.1", "mistralai/Mistral-7B-Instruct-v0.3"),
    ("mistralai/Mistral-7B-Instruct-v0.1_quantized", "mistralai/Mistral-7B-Instruct-v0.3_quantized"),
    ("meta-llama/Llama-2-7b-chat-hf", "meta-llama/Meta-Llama-3-8B-Instruct"),
    ("meta-llama/Llama-2-7b-chat-hf_quantized", "meta-llama/Meta-Llama-3-8B-Instruct"),
]
```

Execute ```python compare_models.py```.


### Full H-Divergence Report
To get a comprehensive H-Divergence report with average H-Divergence and Top/Bottom-5 subjects (including values):


In the file ```full_exp.py```,
Specify compared models and the reference model, e.g.:
```
    model1 = "mistralai/Mistral-7B-Instruct-v0.1_quantized"
    model2 = "mistralai/Mistral-7B-Instruct-v0.1"
    ref_model = "random"
```

Execute ```python full_exp.py```.
