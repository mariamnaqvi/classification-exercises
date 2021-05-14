import pandas as pd
import numpy as np
import os
# use get_db_url function to connect to the codeup db
from env import get_db_url


# ---------------- Acquire Titanic Data ---------------------- #

def get_titanic_data(cached = False):
    '''
    This function returns the titanic database as a pandas dataframe. If the data is cached or the file exists in the directory, the
    function will read the data into a df and return it. Otherwise, the function will read the database into a dataframe, cache it as a csv file
    and return the dataframe.
    '''
    #If the cached parameter is false, or the csv file is not on disk, read from the database into a dataframe
    if cached == False or os.path.isfile('titanic_df.csv') == False:
        query = '''
        SELECT * 
        FROM passengers;
        '''
        titanic_df = pd.read_sql(query, get_db_url('titanic_db'))
        #also cache the data we read from the db, to a file on disk
        titanic_df.to_csv('titanic_df.csv')
    else:
        #either the cached parameter was true, or a file exists on disk. Read that into a df instead of going to the database
        titanic_df = pd.read_csv('titanic_df.csv', index_col=0)

    #return our dataframe regardless of its origin
    return titanic_df
    

# ---------------- Acquire Iris Data ---------------------- #

def get_iris_data(cached = False):
    '''
    This function will return the iris db as a pandas df. If the data is cached or the file exists in the directory, the function
    will read that file into a pandas df and return it. Otherwise, the function will read data from the codeup db into a df,
    and return it to the caller.
    '''

    # read the db from codeup db into a df if the cached parameter is false or the file is not on disk
    if cached == False or os.path.isfile('iris_df.csv') == False:
        query = '''
        SELECT * 
        FROM measurements
        JOIN species USING (species_id);;
        '''
        iris_df = pd.read_sql(query, get_db_url('iris_db'))
        # cache it as a csv file
        iris_df.to_csv('iris_df.csv')

    else: # if cached parameter is True or file exists on disk, read the file into a pandas df
        iris_df = pd.read_csv('iris_df', index_col=0)
     # return the iris df regardless of origin
    return iris_df
