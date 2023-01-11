'''
A module for feature preparation.
'''


def get_df_part(df,part):
    '''
    Get a part of a dataframe that contains only the columns of P-loads, Q-loads or buses.
    Args:
        part (str): can be 'p', 'q' or 'bus'
    '''
    if part=='p':
        df_ret = df.loc[:,df.columns.str.contains('.+_p$')] 
    elif part=='q':
        df_ret = df.loc[:,df.columns.str.contains('.+_q$')] 
    elif part=='bus':
        df_ret = df.loc[:,df.columns.str.contains('^Bus\d+')] 
    return df_ret

