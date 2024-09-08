import pandas as pd
import numpy as np
import compute_hdiv as ch


if __name__ == '__main__':
    df = pd.read_csv(ch.RESULTS_PATH, sep=", ")
    subjects_list = df.columns[1:]
    model1 = "mistralai/Mistral-7B-Instruct-v0.1_quantized"
    model2 = "mistralai/Mistral-7B-Instruct-v0.1"
    ref_model = "random"

    # Print the 5 subjects with the highest h-divergence and the 5 subjects with the lowest h-divergence
    h_divergences = []
    for subject in subjects_list:
        h_divergences.append(ch.h_diveregence(model1, model2, ref_model, subject))
    h_divergences = np.array(h_divergences)
    sorted_indices = np.argsort(h_divergences)
    print('\n=============================\n')
    print("Subjects with the highest h-divergence:")
    for i in range(1, 6):
        print(subjects_list[sorted_indices[-i]], h_divergences[sorted_indices[-i]])
    print('\n=============================\n')
    print("Subjects with the lowest h-divergence:")
    while h_divergences[sorted_indices[0]] == 0.0:
        print(subjects_list[sorted_indices[0]], h_divergences[sorted_indices[0]])
        sorted_indices = sorted_indices[1:]
        if len(sorted_indices) == 0:
            break
        else:
            for i in range(min(5, len(sorted_indices))):
                print(subjects_list[sorted_indices[i]], h_divergences[sorted_indices[i]])
