'''
Unit tests for the functions in churn_library.py

Author: Xinyi Mao
Date: July 23 2025
'''
import logging
import joblib
import pytest
import churn_library as cls



logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture(scope="module")
def df_original():
    '''
    Pytest fixture to pass original dataframe to various tests
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        # logging.info("Testing import_data: SUCCESS")
        return df
    except FileNotFoundError as err:
        # logging.error("Testing import_eda: The file wasn't found")
        raise err


@pytest.fixture(scope="module")
def response():
    '''
    Pytest fixture to pass response to various tests
    '''
    return 'Churn'


@pytest.fixture(scope="module")
def df_with_esponse(df_original, response):
    '''
    Pytest fixture to pass dataframe with response
    '''
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    encoded_df = cls.encoder_helper(df_original, cat_columns, response)
    return encoded_df




@pytest.fixture(scope="module")
def df_engineered(df_with_esponse, response):
    '''
    Pytest fixture to pass feature engineered dataset splits for training
    '''
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
            df_with_esponse, response)
    return X_train, X_test, y_train, y_test


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(df_original):
    '''
    test perform eda function
    '''
    cls.perform_eda(df_original)
    for image in ['churn_distribution',
                  'customerAge_distribution',
                  'maritalStatus_distribution',
                  'totalTransCt_distribution',
                  'correlation_heatmap']:
        try:
            with open(f'./images/eda/{image}.png', "r", encoding='utf-8'):
                logging.info(
                    'Testing eda: ./images/eda/%s.png successfully open',
                    image)
        except FileNotFoundError as err:
            logging.error(
                'Testing eda Error: ./images/eda/%s.png missing',
                image)
            raise err


def test_encoder_helper(df_original, response):
    '''
    test encoder helper
    '''
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    try:
        encoded_df = cls.encoder_helper(df_original, cat_columns, response)
        for col in cat_columns:

            assert col + '_' + \
                response in encoded_df.columns, f'{col}_{response} column is missing from the DataFrame'
        verify_value = encoded_df['Gender_Churn'].iloc[0]
        assert verify_value == 0.14615223317257287, f'{verify_value} is not the right value which should be 0.146152'
        logging.info('Testing ecoder_helper: SUCCESS')
    except AssertionError as err:
        logging.error(err)
        raise err


def test_perform_feature_engineering(df_withResponse, response):
    '''
    test perform_feature_engineering
    '''
    try:
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
            df_withResponse, response)
        assert X_train.shape[0] > 0, 'X_train does not have rows'
        assert X_test.shape[0] > 0, 'X_test does not have rows'
        assert y_train.shape[0] > 0, 'y_train does not have rows'
        assert y_test.shape[0] > 0, 'y_test does not have rows'
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(err)
        raise err


def test_train_models(df_engineered):
    '''
    test train_models
    '''
    cls.train_models(
        df_engineered[0],
        df_engineered[1],
        df_engineered[2],
        df_engineered[3])
    # test if report result saved
    labels = ['train', 'test']
    for label in labels:
        # logistic regression
        try:
            with open(f'./images/results/classfication_report_lr_{label}.png',
                       "r",encoding='utf-8'):
                logging.info(
                    'Testing train_models report saved: '
					'./images/results/classfication_report_lr_%s.png successfully open',
                    label)
        except FileNotFoundError as err:
            logging.error(
                'Testing train_models report saved: '
				'./images/results/classfication_report_lr_%s.png missing',
                label)
            raise err

        # random forest
        try:
            with open(f'./images/results/classfication_report_rf_{label}.png',
                       "r",encoding='utf-8'):
                logging.info(
                    'Testing train_models report saved: '
					'./images/results/classfication_report_rf_%s.png successfully open',
                    label)
        except FileNotFoundError as err:
            logging.error(
                'Testing train_models report saved: '
				'./images/results/classfication_report_rf_%s.png missing',
                label)
            raise err

    # test if feature importance plot saved
    try:
        with open('./images/results/feature_importance_plot.png', "r",encoding='utf-8'):
            logging.info('Testing train_models feature importance plot saved')
    except FileNotFoundError as err:
        logging.error('Testing train_models feature importance plot missing')
        raise err

    # test if roc plot is saved
    try:
        with open('./images/results/roc_curves.png', "r",encoding='utf-8'):
            logging.info('Testing train_models roc curve plot saved')
    except FileNotFoundError as err:
        logging.error('Testing train_models roc curve plot missing')
        raise err

    # test if the rf model is save
    try:
        joblib.load('./models/rfc.pkl')
        logging.info('Testing train_models rf model saved')
    except FileNotFoundError as err:
        logging.error('Testing train_models rf model not saved')

    # test if the lr model is save
    try:
        joblib.load('./models/logistic_model.pkl')
        logging.info('Testing train_models lr model saved')
    except FileNotFoundError as err:
        logging.error('Testing train_models lr model not saved')


if __name__ == "__main__":
    pytest.main(["churn_script_logging_and_tests.py"])
