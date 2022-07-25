import sys, os
import pprint

def get_preprocess_params(data_schema): 
    # initiate the pp_params dict
    pp_params = {}   
            
    # set the id attribute
    pp_params["id_field"] = data_schema["inputDatasets"]["recommenderBaseMainInput"]["idField"]   
            
    # set the user_id attribute
    pp_params["user_field"] = data_schema["inputDatasets"]["recommenderBaseMainInput"]["userField"]   
    
    # set the item_id attribute
    pp_params["item_field"] = data_schema["inputDatasets"]["recommenderBaseMainInput"]["itemField"]
    
    # set the target attribute
    pp_params["target_field"] = data_schema["inputDatasets"]["recommenderBaseMainInput"]["targetField"]   
    
    # pprint.pprint(pp_params)   ; sys.exit()  
    return pp_params


