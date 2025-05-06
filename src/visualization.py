import seaborn as sns
import matplotlib.pyplot as plt

def plot_depression_distribution(df):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Depression', data=df, palette=['blue', 'red'])
    plt.title('Distribution of Depression')
    plt.xlabel('Depressed (0 = No, 1 = Yes)')
    plt.ylabel('Count')
    plt.show()

def plot_correlation_heatmap(df):
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.1, annot_kws={'size': 8})
    plt.title('Correlation Heatmap of all Features')
    plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

def plot_model_comparison(models, accuracies):
    plt.figure(figsize=(8, 6))
    plt.bar(models, accuracies, color=['blue', 'green', 'red'])
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])

    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, str(round(v, 4)), ha='center')

    plt.savefig('outputs/figures/model_accuracy_comparison.png')

    plt.show()


    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, str(round(v, 4)), ha='center')

    plt.show()
