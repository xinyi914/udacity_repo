# library doc string
'''
Library of functions to find customers who are likely to churn

Author: Xinyi Mao
Date: July 23 2025
'''

# import libraries
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import RocCurveDisplay, classification_report
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''

    # plot distribution of churn
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.title('Distribution of Churn')
    plt.xlabel('Churn')
    plt.ylabel('Frequency')
    plt.savefig(
        './images/eda/churn_distribution.png',
        dpi=300,
        bbox_inches='tight')
    plt.close()

    # plot distribution of customer age
    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.title('Distribution of Customer Age')
    plt.xlabel('Customer Age')
    plt.ylabel('Frequency')
    plt.savefig(
        './images/eda/customerAge_distribution.png',
        dpi=300,
        bbox_inches='tight')
    plt.close()

    # plot distribution of marital status
    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.title('Marital Status Distribution (Normalized)')
    plt.xlabel('Marital Status')
    plt.ylabel('Frequency')
    plt.savefig(
        './images/eda/maritalStatus_distribution.png',
        dpi=300,
        bbox_inches='tight')
    plt.close()

    # plot distribution of total trans ct
    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.title('Distribution of Total_Trans_Ct')
    plt.xlabel('Total_Trans_Ct')
    plt.ylabel('Distribution')
    plt.savefig(
        './images/eda/totalTransCt_distribution.png',
        dpi=300,
        bbox_inches='tight')
    plt.close()

    # plot heatmap of the correlation of variables
    plt.figure(figsize=(20, 10))
    sns.heatmap(
        df.select_dtypes(
            include='number').corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    plt.title('Heatmap of correlations')
    plt.savefig(
        './images/eda/correlation_heatmap.png',
        dpi=300,
        bbox_inches='tight')
    plt.close()


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
            used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    df[response] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    for cat_col in category_lst:
        new_col_name = cat_col + '_' + response
        df[new_col_name] = df[cat_col].map(
            df.groupby(cat_col)[response].mean())
    return df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that
              could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    y = df[response]
    x_variables = pd.DataFrame()
    df_encoded = encoder_helper(df, cat_columns, response)
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    x_variables[keep_cols] = df_encoded[keep_cols]
    x_train, x_test, y_train, y_test = train_test_split(
        x_variables, y, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    labels = ['train', 'test']
    lr_results = [y_train_preds_lr, y_test_preds_lr]
    rf_results = [y_train_preds_rf, y_test_preds_rf]
    y_lists = [y_train, y_test]
    # plot training classification report
    for idx in range(2):
        label = labels[idx]
        lr_result = lr_results[idx]
        y = y_lists[idx]
        lr_report = classification_report(y, lr_result, output_dict=True)
        report_df = pd.DataFrame(lr_report).transpose()

        plt.figure(figsize=(10, 6))
        plt.axis('off')
        table = plt.table(cellText=report_df.round(2).values,
                          colLabels=report_df.columns,
                          rowLabels=report_df.index,
                          loc='center',
                          cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        plt.title(f'./images/results/classfication_report_lr_{label}')
        plt.savefig(
            f'./images/results/classfication_report_lr_{label}.png',
            bbox_inches='tight',
            dpi=300)
        plt.close()

    # plot logistic regression test classification report
    for idx in range(2):
        label = labels[idx]
        rf_result = rf_results[idx]
        y = y_lists[idx]
        rf_report = classification_report(y, rf_result, output_dict=True)
        report_df = pd.DataFrame(rf_report).transpose()

        plt.figure(figsize=(10, 6))
        plt.axis('off')
        table = plt.table(cellText=report_df.round(2).values,
                          colLabels=report_df.columns,
                          rowLabels=report_df.index,
                          loc='center',
                          cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        plt.title(f'./images/results/classfication_report_rf_{label}')
        plt.savefig(
            f'./images/results/classfication_report_rf_{label}.png',
            bbox_inches='tight',
            dpi=300)
        plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''

    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # store roc curve plot
    lrc_plot = RocCurveDisplay.from_estimator(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = RocCurveDisplay.from_estimator(
        cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.title('ROC curve of logistic regresision and random forest')
    plt.savefig('./images/results/roc_curves.png')
    plt.close()

    # store model report result
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    # store feature importance plot
    feature_importance_plot(
        cv_rfc,
        X_train,
        './images/results/feature_importance_plot.png')

    # save the model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


if __name__ == '__main__':
    path = "./data/bank_data.csv"
    df = import_data(path)
    perform_eda(df)
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, 'Churn')
    train_models(X_train, X_test, y_train, y_test)
