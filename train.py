from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import json
import dagshub
from xgboost import XGBClassifier

# Load data 
df = pd.read_csv("heartDisease.csv")

y = df['num']
x = df.drop(['num'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

def main():
    model = XGBClassifier().fit(x_train, y_train)
    
    train_accuracy = model.score(x_train, y_train)
    test_accuracy = model.score(x_test, y_test)
    with open('metrics.json','w') as of:
        json.dump({ "accuracy": test_accuracy}, of)
        of.close()
    with dagshub.dagshub_logger() as logger:
        logger.log_hyperparams(model_class=type(model).__name__)
        logger.log_hyperparams({'model': model.get_params()})
        logger.log_metrics({f'accuracy':round(test_accuracy,3)})
    importance = model.feature_importances_
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance
    plt.barh(x.columns, importance)
    plt.savefig('feature_importance.png')

if __name__ == '__main__':
    main()
