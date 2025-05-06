from preprocessing import preprocess_data
from modeling import train_logistic_regression, train_naive_bayes, train_decision_tree, evaluate_model
from visualization import plot_depression_distribution, plot_correlation_heatmap, plot_model_comparison

file = file = 'E:/CSE422/student_depression_analysis/data/Student_Depression_Dataset.csv'
df = preprocess_data(file)
print("Preprocessing done.")

X = df.drop('Depression', axis=1)
y = df['Depression']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

log_reg_model = train_logistic_regression(X_train, y_train)
naive_bayes_model = train_naive_bayes(X_train, y_train)
dt_model = train_decision_tree(X_train, y_train)

log_reg_accuracy, _, log_reg_report = evaluate_model(log_reg_model, X_test, y_test)
print("Logistic Regression Accuracy:", log_reg_accuracy)

naive_bayes_accuracy, _, nb_report = evaluate_model(naive_bayes_model, X_test, y_test)
print("Naive Bayes Accuracy:", naive_bayes_accuracy)

dt_accuracy, _, dt_report = evaluate_model(dt_model, X_test, y_test)
print("Decision Tree Accuracy:", dt_accuracy)

models = ['Logistic Regression', 'Naive Bayes', 'Decision Tree']
accuracies = [log_reg_accuracy, naive_bayes_accuracy, dt_accuracy]
plot_model_comparison(models, accuracies)

