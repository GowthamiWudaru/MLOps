from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import json
import dagshub

# Load data 
df = pd.read_csv("heartDisease.csv")

y = df['num']
x = df.drop(['num'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=50, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=400, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    print("Regularization Strength:", np.float(args.C))
    print("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
    
    os.makedirs('outputs',exist_ok=True)
    joblib.dump(model,'outputs/LogisticRegression.pkl')
    
    train_accuracy = model.score(x_train, y_train)
    test_accuracy = model.score(x_test, y_test)
    with open('metrics.json','w') as of:
        json.dump({ "accuracy": test_accuracy}, of)
        of.close()
    with dagshub.dagshub_logger() as logger:
        logger.log_hyperparams(model_class=type(model).__name__)
        logger.log_hyperparams({'model': model.get_params()})
        logger.log_metrics({f'accuracy':round(test_accuracy,3)})
    importance = model.coef_
    for i,v in enumerate(importance):
        print(i,v)
        print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.savefig('feature_importance.png')

if __name__ == '__main__':
    main()
