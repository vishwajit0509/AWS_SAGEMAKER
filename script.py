## saving this model to the s3 bucket
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,precision_score

import sklearn
import joblib
import boto3
import pathlib
from io import StringIO
import argparse
import joblib
import os
import numpy as np
import pandas as pd

## to push the model in the s3
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

if __name__ == "__main__":
    print("[INFO] Extracting arguments")
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=0)

    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train-file", type=str, default="train-V-1.csv")  # Corrected to uppercase "V"
    parser.add_argument("--test-file", type=str, default="test-V-1.csv")    # Already correct

    args, _ = parser.parse_known_args()

    print("SKLearn Version: ",sklearn.__version__)
    print("Joblib Version: ",joblib.__version__)

    print("[INFO] Reading data")

    print()

    # Load data
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))


    features = list(train_df.columns)
    label = features.pop(features.index('price_range'))

    print("Building training and testing datasets")

    print()

    X_train = train_df[features]
    X_test = test_df[features]
    y_train = train_df[label]
    y_test = test_df[label]


    print('Column order : ')

    print(features)
    print()

    print("Label column is: ",label)
    print()

    print("Data Shape : ")
    print()

    print("---- SHAPE OF TRAINING DATA (85%) ----")
    print(X_train.shape)
    print(y_train.shape)

    print()

    print("----SHAPE OF TESTING DATA(15%)----")
    print(X_test.shape)
    print(y_test.shape)
    print()

    print("Training RandomForest Model.....")
    print()

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state
    )
    model.fit(X_train, y_train)

    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)

    print("Model persisted at "+model_path)
    print()

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    print()
    print("---- METRIS RESULTS FOR TESTING DATA----")
    print()

    print("Total Rows are:",X_test.shape[0])

    ##print('[TESTING] Model Accuracy is: ',test_acc)
    ##print('[Testing] Testing Report: ')
    ##print(test_rep)



        

