import numpy as np
import pandas as pd
from scipy.optimize import minimize


class Linear_Regressor:
    
    ###############################################
    
    '''
    
    Leave this one unchanged except for the model name!
    
    '''
    
    def __init__(self):
        self.needs_training = True 
        self.name = 'AR1'
    
    ##############################################
    
    '''
    
    train_df will be a dataframe, with rows sorted in chronological order (most recent last).
    
    The dataframe will be of the form:
    
    Region |  Week  |  Cases
    __________________________
    Oxford |   23   |  19284
    
    
    Note that the week simply refers to the week of the year (to allow for seasonality to be incorporated).
    
    
    ''' 
    
    
    def train(self, train_df):
        self.needs_training = False ### Keep this here!
        
        ###### Process Data (we want an array with current, last week, and two weeks ago cases)
        
        
        split_df = [group for _, group in train_df.groupby('Region')]


        processed_data = np.zeros((len(train_df)-2*len(split_df),3))

        curr_row = 0
        for df in split_df:
            df['Back1'] = df['Cases'].shift(periods = 1)
            df['Back2'] = df['Cases'].shift(periods = 2)
            df = df.dropna()
            processed_data[curr_row:curr_row + len(df)] = df[['Cases','Back1','Back2']].to_numpy()
            curr_row += len(df)

            
        #### Our model is current = c[0]*last_week + c[1]*two_weeks_ago
            
        def Loss(c):

            return np.sum(np.square(c[0]*processed_data[:,1] + c[1]*processed_data[:,2] - processed_data[:,0]))

        #### Store trained parameter for prediction function
        
        self.c = minimize(Loss,np.ones(2)).x 
    
    '''
    
    There should be no outputs from train_df
    
    '''
    
    
    ##############################################
    
    
    '''
    
    test_df and all_df will be the same structure as train_df
    
    '''
    
    
    def predict(self, test_df,all_df,weeks_ahead):

        all_df['Ind'] = np.arange(len(all_df)) ### Add index so we can map back our predictions later


        ###### Process Data (we want an array with current, last week, and two weeks ago cases)

        split_df = [group for _, group in all_df.groupby('Region')]


        processed_data = np.zeros((len(all_df)-len(split_df),3))

        curr_row = 0
        for df in split_df:
            df['Back1'] = df['Cases'].shift(periods = 1)
            df = df.dropna()
            processed_data[curr_row:curr_row + len(df)] = df[['Ind','Back1','Cases']].to_numpy()
            
            
            
            curr_row += len(df)




        #### Perform Forecasts

        outputs_temp = np.zeros((len(processed_data),weeks_ahead+3))
        outputs_temp[:,:3] = processed_data
        for ahead in range(3,3+weeks_ahead):
            outputs_temp[:,ahead] = self.c[1]*outputs_temp[:,ahead-2] + self.c[0]*outputs_temp[:,ahead-1]

        #print(outputs_temp[:,1:])
        #print(outputs_temp[:,0])
        #### Final Outputs (on all data)

        outputs_final = np.zeros((len(all_df),weeks_ahead))
        outputs_final[outputs_temp[:,0].astype(int)] = outputs_temp[:,3:]


        ##### Trim to test data
        columns = []
        for ahead in range(weeks_ahead):
            all_df['Prediction ' + str(ahead+1)] = outputs_final[:,ahead]
            columns.append('Prediction ' + str(ahead+1))

        test_df = pd.merge(test_df,all_df[columns],left_index=True,right_index=True)
        
        return test_df
    
    '''
    
    The outputs should be a numpy vector of size (len(test_df),weeks_ahead) 
    
    '''
    
    ################################################
    def name(self):
        
        return model.name
    