
def check_or_make_var(meta_data,key_name,get_default_val_func=lambda :[]):
    if meta_data.get(key_name,None) is None:
        meta_data[key_name]=get_default_val_func()

        