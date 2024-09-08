import pandas as pd
import numpy as np
RESULTS_PATH = "results.csv"

df = pd.read_csv(RESULTS_PATH, sep=", ")

def h_diveregence(model1, model2, ref_model, subject):
    def h_entropy(model, ref_model):
        subject_column = df.columns.get_loc(subject)
        model_loss = df[df["hf_model_path"] == model].values[0, subject_column]
        if ref_model == 'random':
            ref_loss = 0.75
        else:
            ref_loss = df[df["hf_model_path"] == ref_model].values[0, subject_column]
        return min(np.sum(model_loss), np.sum(ref_loss))

    def h_entropy_mixed(ref_model):
        subject_column = df.columns.get_loc(subject)
        model_loss1 = df[df["hf_model_path"] == model1].values[0, subject_column]
        model_loss2 = df[df["hf_model_path"] == model2].values[0, subject_column]
        mixed_loss = 0.5*(model_loss1+model_loss2)
        if ref_model == 'random':
            ref_loss = 0.75
        else:
            ref_loss = df[df["hf_model_path"] == ref_model].values[0, subject_column]
        return min(np.sum(mixed_loss), np.sum(ref_loss))
    
    return h_entropy_mixed(ref_model) - min(h_entropy(model1, ref_model), h_entropy(model2, ref_model))
