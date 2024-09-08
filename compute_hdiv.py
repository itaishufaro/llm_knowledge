import pandas as pd
import numpy as np
RESULTS_PATH = "results.csv"

df = pd.read_csv(RESULTS_PATH, sep=",")

def h_diveregence(model1, model2, ref_model, subject):
    def h_entropy(model, ref_model):
        loss_column = df.columns.get_loc('mean_loss')
        model_loss = df[(df['hf_model_path'] == model) & (df['mmlu_subject'] == subject)].values[0, loss_column]
        if ref_model == 'random':
            ref_loss = 0.75
        else:
            ref_loss = df[(df['hf_model_path'] == ref_model) & (df['mmlu_subject'] == subject)].values[0, loss_column]
        return min(np.sum(model_loss), np.sum(ref_loss))

    def h_entropy_mixed(ref_model):
        loss_column = df.columns.get_loc('mean_loss')
        model_loss1 = df[(df['hf_model_path'] == model1) & (df['mmlu_subject'] == subject)].values[0, loss_column]
        model_loss2 = df[(df['hf_model_path'] == model2) & (df['mmlu_subject'] == subject)].values[0, loss_column]
        mixed_loss = 0.5*(model_loss1+model_loss2)
        if ref_model == 'random':
            ref_loss = 0.75
        else:
            ref_loss = df[(df['hf_model_path'] == ref_model) & (df['mmlu_subject'] == subject)].values[0, loss_column]
        return min(np.sum(mixed_loss), np.sum(ref_loss))
    
    return h_entropy_mixed(ref_model) - min(h_entropy(model1, ref_model), h_entropy(model2, ref_model))
