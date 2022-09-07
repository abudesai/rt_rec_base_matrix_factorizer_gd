import numpy as np, pandas as pd
import sys 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


class UserItemIdMapper(BaseEstimator, TransformerMixin):    
    ''' Generates sequential user and item ids for internal use.'''
    def __init__(self, user_id_col, item_id_col, user_id_int_col, item_id_int_col): 
        super().__init__()
        self.user_id_col = user_id_col
        self.user_id_int_col = user_id_int_col
        self.item_id_col = item_id_col
        self.item_id_int_col = item_id_int_col
        self.new_to_orig_user_map = None
        self.new_to_orig_item_map = None

    
    def fit(self, data): 

        self.user_ids = data[[self.user_id_col]].drop_duplicates()

        # self.user_ids = self.user_ids.sample(n=1000, replace=False, random_state=42)        
        
        self.user_ids[self.user_id_int_col] = self.user_ids[self.user_id_col].factorize(na_sentinel=None)[0]
        
        self.users_orig_to_new = dict( zip(self.user_ids[self.user_id_col], 
            self.user_ids[self.user_id_int_col]) )   

        self.item_ids = data[[self.item_id_col]].drop_duplicates()        
        
        self.item_ids[self.item_id_int_col] = self.item_ids[self.item_id_col].factorize(na_sentinel=None)[0]

        self.items_orig_to_new = dict( zip(self.item_ids[self.item_id_col], 
            self.item_ids[self.item_id_int_col]) )
        
        self.users_new_to_orig = { v:k for k,v in self.users_orig_to_new.items()}
        self.items_new_to_orig = { v:k for k,v in self.items_orig_to_new.items()}      

        return self


    def transform(self, df): 
        idx1 = df[self.user_id_col].isin(self.users_orig_to_new.keys())
        idx2 = df[self.item_id_col].isin(self.items_orig_to_new.keys())
        df = df.loc[idx1 & idx2].copy()        

        df[self.user_id_int_col] = df[self.user_id_col].map(self.users_orig_to_new)
        df[self.item_id_int_col] = df[self.item_id_col].map(self.items_orig_to_new)
        
        return df


    def inverse_transform(self, df): 
        df.sort_values(by=[self.user_id_int_col, self.item_id_int_col], inplace=True)
        df[self.user_id_col] = df[self.user_id_int_col].map(self.users_new_to_orig)
        df[self.item_id_col] = df[self.item_id_int_col].map(self.items_new_to_orig)
        return df



class TargetScaler(BaseEstimator, TransformerMixin):  
    ''' Scale target '''
    def __init__(self, target_col, target_int_col, scaler_type='minmax'): 
        super().__init__()
        self.target_col = target_col
        self.target_int_col = target_int_col

        if scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            raise Exception(f"Undefined scaler type {scaler_type}")


    def fit(self, data): 
        self.scaler.fit(data[[self.target_col]])
        return self
        

    def transform(self, data):
        if data.empty: return data
        if not self.target_col in data.columns: return data                
        data[self.target_int_col] = self.scaler.transform(data[[self.target_col]])           
        return data


    def inverse_transform(self, data): 
        return self.scaler.inverse_transform(data)




class XYSplitter(BaseEstimator, TransformerMixin): 
    def __init__(self, id_col, user_int_col, item_int_col, ratings_int_col):
        self.id_col = id_col
        self.user_int_col = user_int_col
        self.item_int_col = item_int_col
        self.ratings_int_col = ratings_int_col
    
    def fit(self, data): return self
    
    def transform(self, data):  
                
        X = data[[self.user_int_col, self.item_int_col]].values
        
        ids = data[self.id_col].values
        
        if self.ratings_int_col in data.columns: 
            y = data[self.ratings_int_col].values
        else: y = None
                
        return { 'X': X, 'y': y, 'ids': ids }