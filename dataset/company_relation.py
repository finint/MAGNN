import os
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

COMPANY_RELATION_DF_PATH = os.path.join(DATA_DIR, 'example_company_relation.pkl')

def get_relation_company(sec_name, relation_df):
    try:
        result = relation_df[relation_df['relevant_companies'].str.contains(sec_name)]
        result = result['relevant_companies']
        if len(result) == 0:
            return None
        result = '|'.join(result.to_list())
        return result
    except Exception as e:
        print(e)

def company_name_to_secu(relation, all_company_info):
    # try:
        # if np.isnan(relation):
    if relation is None:
        return None
    else:
        relation_list = relation.split('|')
        relation_kdcodes = all_company_info[all_company_info['sec_name'].isin(relation_list)]['kdcode'].to_list()
        if len(relation_kdcodes) == 0:
            return None
        kdcode = '|'.join(all_company_info[all_company_info['sec_name'].isin(relation_list)]['kdcode'].to_list())
        return kdcode


def get_count(x):
    # try:
        # if np.isnan(x):
    if x is None:
        return 0
    else:
        relation_list = x.split('|')
        return len(list(set(relation_list)))


def get_company_relation():
    if os.path.exists(COMPANY_RELATION_DF_PATH):
        df = pd.read_pickle(COMPANY_RELATION_DF_PATH)
        return df
    
    print('company relation file DOES NOT exists! Load your file from your database')
    return pd.DataFrame()
