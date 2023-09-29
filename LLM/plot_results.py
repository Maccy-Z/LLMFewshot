import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import numpy as np

LINE_TYPES = {
    'Ordered': '-',
    'Onehot': '--',
    'Biased': '-.',
    'Monotonic': ':',
    'As-is': '-',
}

MODEL_COLORS = {
    'LR': '#B2DF8A',
    'LightGBM': '#A6CEE3',
    'XGBoost': '#FF7F00',
    'TabPFN': '#CAB2D6',
    'TabLLM': '#FFDC00',
    'Biased': '#E41A1C',
    'Monotonic': '#E41A1C',
}

def plot_lines(df, dataset_name, shots=4):
    all_shots = [4, 8, 16, 32, 64, 128, 256, 512]
    if shots not in all_shots:
        raise Exception(f'{shots} is not a valid number of shots. Please select a value from [4, 8, 16, 32, 64, 128, 256, 512].')

    dataset_filtered = df[df['Dataset'] == dataset_name]
    dataset_filtered = dataset_filtered[dataset_filtered['Type'] != 'Raw']
    shot_columns = ['4-shot-auc', '8-shot-auc', '16-shot-auc', '32-shot-auc', '64-shot-auc', '128-shot-auc', '256-shot-auc', '512-shot-auc']
    index = shot_columns.index(f'{shots}-shot-auc')
    shot_columns = shot_columns[index:]
    all_shots = all_shots[index:]


    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    for model, model_data in dataset_filtered.groupby('Model'):
        for type, type_data in model_data.groupby('Type'):
            auc_values = type_data[shot_columns].values[0]  # Extract AUC values
            line_style = LINE_TYPES.get(type, '-')
            if type == 'Biased' or type == 'Monotonic':
                color = MODEL_COLORS.get(type, 'green')
            else:
                color = MODEL_COLORS.get(model, 'green') 
            plt.plot(shot_columns, auc_values, label=f'{model} - {type}', linestyle=line_style, marker='o', color=color)

    # Customize the plot
    plt.xlabel('Number of Shots')
    plt.ylabel('ROC AUC')
    x = range(len(all_shots))
    plt.xticks(x, all_shots)
    plt.title(f'ROC AUC Comparison for {dataset_name} Dataset')
    plt.legend(loc='best')

    # Show the plot or save it to a file
    plt.tight_layout()
    plt.savefig(f'./results/{dataset_name}_{shots}-shots_results.png')

if __name__ == "__main__":
    df = pd.read_csv('../results.csv')

    

    for ds_name in df['Dataset'].unique():
        plot_lines(df, dataset_name=ds_name)