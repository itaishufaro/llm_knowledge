import pandas as pd
import numpy as np
import compute_hdiv as ch


if __name__ == '__main__':
    df = pd.read_csv(ch.RESULTS_PATH, sep=",")
    subjects_list = df['mmlu_subject'].unique()
    model1 = "meta-llama/Meta-Llama-3-8B-Instruct"
    model2 = "mistralai/Mistral-7B-Instruct-v0.3"
    ref_model = "random"

    # Print the 5 subjects with the highest h-divergence and the 5 subjects with the lowest h-divergence
    h_divergences = []
    for subject in subjects_list:
        h_divergences.append(ch.h_diveregence(model1, model2, ref_model, subject))
    h_divergences = np.array(h_divergences)
    sorted_indices = np.argsort(h_divergences)
    print("Subjects with the lowest h-divergence:")
    for i in range(5):
        print(subjects_list[sorted_indices[i]], h_divergences[sorted_indices[i]])
    print("Subjects with the highest h-divergence:")
    for i in range(1, 6):
        print(subjects_list[sorted_indices[-i]], h_divergences[sorted_indices[-i]])