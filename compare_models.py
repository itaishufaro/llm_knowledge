import numpy as np
import pandas as pd
import compute_hdiv as ch

RESULTS_PATH = "results.csv"
OUTPUT_PATH = "out.csv"

ref_model = "random"

df = pd.read_csv(RESULTS_PATH, sep=", ")
subjects_list = df.columns[1:-1]

top_k = 5
force_lowest_positive_hdiv = False

models_list = df["hf_model_path"].tolist()
quantized_models = [model for model in models_list if model+'_quantized' in models_list]

comparisons = [
    ("mistralai/Mistral-7B-Instruct-v0.1", "mistralai/Mistral-7B-Instruct-v0.3"),
    ("mistralai/Mistral-7B-Instruct-v0.1_quantized", "mistralai/Mistral-7B-Instruct-v0.3_quantized"),
    ("meta-llama/Llama-2-7b-chat-hf", "meta-llama/Meta-Llama-3-8B-Instruct"),
    ("meta-llama/Llama-2-7b-chat-hf_quantized", "meta-llama/Meta-Llama-3-8B-Instruct"),

]

# Sizes comparisons
# comparisons = [
#     ("meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-70B-Instruct"),
#     ("meta-llama/Meta-Llama-3-8B-Instruct_quantized", "meta-llama/Meta-Llama-3-70B-Instruct_quantized"),
#     ("microsoft/Phi-3.5-mini-instruct", "microsoft/Phi-3.5-MoE-instruct"),
#     ("microsoft/Phi-3.5-mini-instruct_quantized", "microsoft/Phi-3.5-MoE-instruct_quantized"),
#     ("google/gemma-1.1-2b-it", "google/gemma-1.1-7b-it"),
#     ("google/gemma-1.1-2b-it_quantized", "google/gemma-1.1-7b-it_quantized")
# ]


if __name__ == '__main__':
    results = []
    for model1, model2 in comparisons:
        avg_h_div = ch.h_diveregence(model1, model2, ref_model, 'avg')
        h_divergences = []
        for subject in subjects_list:
            h_divergences.append(ch.h_diveregence(model1, model2, ref_model, subject))
        h_divergences = np.array(h_divergences)
        sorted_indices = np.argsort(h_divergences)
        highest_h_div = [subjects_list[sorted_indices[-i]] for i in range(1, top_k+1)]
        if force_lowest_positive_hdiv:
            zero_h_div = []
            while h_divergences[sorted_indices[0]] == 0.0:
                zero_h_div.append(subjects_list[sorted_indices[0]])
                sorted_indices = sorted_indices[1:]
                if len(sorted_indices) == 0:
                    break
            lowest_positive_h_div = [subjects_list[sorted_indices[i]] for i in range(min(top_k, len(sorted_indices)))]
            results.append((model1.split('/')[1], model2.split('/')[1], round(avg_h_div, 3), highest_h_div, zero_h_div, lowest_positive_h_div))
        else:
            lowest_h_div = [subjects_list[sorted_indices[i]] for i in range(top_k)]
            results.append((model1.split('/')[1], model2.split('/')[1], round(avg_h_div, 3), highest_h_div, lowest_h_div))

    if force_lowest_positive_hdiv:
        results_df = pd.DataFrame(results, columns=["Model 1", "Model 2", "Average H-Divergence", "Subjects with the highest h-divergence", "Subjects with 0 h-divergence", "Subjects with the lowest positive h-divergence"])
    else:
        results_df = pd.DataFrame(results, columns=["Model 1", "Model 2", "Average H-Divergence", "Subjects with the highest h-divergence", "Subjects with the lowest h-divergence"])
    results_df.to_csv(OUTPUT_PATH, index=False)
    