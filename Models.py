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
    
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson
from scipy.stats import norm 
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


class CMJ:
    
    ###############################################
    
    '''
    
    Leave this one unchanged except for the model name!
    
    '''
    
    def __init__(self):
        self.needs_training = True 
        self.name = 'CMJ'
    
    ##############################################
    
    '''
    
    train_df will be a dataframe, with rows sorted in chronological order (most recent last).
    
    The dataframe will be of the form:
    
    Region |  Week  |  Cases
    __________________________
    Oxford |   23   |  19284
    
    
    Note that the week simply refers to the week of the year (to allow for seasonality to be incorporated).
    
    
    ''' 
    def Likelihood(self,data,params):
        rho = np.abs(params[:-1])
        sigma = np.abs(params[-1])
        like_poiss = np.sum(poisson.logpmf(data[1:],data[:-1]*rho))
        like_rho = np.sum(norm.logpdf((rho[1:]-rho[:-1]),scale=sigma))
        return like_poiss + like_rho
    
    def fit_data(self,data):
        out = np.zeros((1200,len(data)))
        likes = np.zeros(1200)
        params = np.ones(len(data))
        curr_like = self.Likelihood(data,params)
        for trial in tqdm(range(1200)):
            for m in range(200):
                params_trial = params + 0.01*np.random.randn(len(data))
                like_trial = self.Likelihood(data,params_trial)
                if like_trial - curr_like > np.log(np.random.random()):
                    params = np.copy(params_trial)
                    curr_like = like_trial + 0
            out[trial] = params
            likes[trial] = curr_like
        return out[200:,-2:]


    def predict_single(self,rhos,weeks_ahead,sigmas,datum_in):
        cases = np.zeros((4000,weeks_ahead))

        for n in range(1000):
            for m in range(4):
                curr_cases = datum_in+0
                r = rhos[n]
                sigma = sigmas[n]
                for week in range(weeks_ahead):
                    r = max(0.25,r + np.random.randn()*sigma)
                    cases[n*4 + m,week] = poisson.rvs(mu = r*curr_cases)

        return cases
    
    def Update_Rho(self,current_rhos,previous_cases,previous_previous_cases):
        
        def Like(rho_new):
            return np.sum(poisson.logpmf(previous_cases,previous_previous_cases*rho_new)) + np.sum(norm.logpdf((rho_new - current_rhos),scale=sigmas))



        rho_new = current_rhos + 0
        curr_like = Like(rho_new)
        for burnin in range(1000):
            rho_trial = rho_new + 0.01*np.random.randn(len(current_rhos))
            like_trial = Like(rho_trial)
            if like_trial - curr_like > np.log(np.random.random()):
                rho_new = np.copy(rho_trial)
                curr_like = like_trial + 0


        rhos = np.zeros(100000)

        for sample in range(10):
            for m in range(100):
                rho_trial = current_rhos + 0.01*np.random.randn(len(current_rhos))
                like_trial = Like(rho_trial)
                if like_trial - curr_like > np.log(np.random.random()):
                    rho_new = np.copy(rho_trial)
                    curr_like = like_trial + 0
            rhos[sample*1000:(sample+1)*1000] = rho_new



        current_rhos = rhos[np.argsort(np.random.random(10000)).astype(int)[:1000]]
        return current_rhos
    

    def train(self, train_df):
        self.needs_training = False ### Keep this here!
        
        ###### Process Data (we want an array with current, last week, and two weeks ago cases)
        
        
        split_df = [group for _, group in train_df.groupby('Region')]


        cases = np.zeros((len(split_df),2))

        
        for n,df in tqdm(enumerate(split_df)):
            data = df['Cases'].to_numpy().astype(int)
            data[data == 0] = 1
            outputs = self.fit_data(data)
            cases[n] = [data[-2],data[-1]]
            try:
                output_df[np.unique(df['Region'])[0] + ' Rhos'] = outputs[:,-2]
                output_df[np.unique(df['Region'])[0] + ' Sigmas'] = outputs[:,-1]
                
                
            except:
                output_df = pd.DataFrame(outputs,columns = [np.unique(df['Region'])[0] + ' Rhos',
                                                           np.unique(df['Region'])[0] + ' Sigmas'])
                
                
            
        self.output_df = output_df
        self.cases = cases
        return output_df
       
            
            

            
    
    '''
    
    There should be no outputs from train_df
    
    '''
    
    
    ##############################################
    
    
    '''
    
    test_df and all_df will be the same structure as train_df
    
    '''
    
    
    def predict(self, test_df,all_df,weeks_ahead):

        test_df['Ind'] = np.arange(len(test_df)) ### Add index so we can map back our predictions later

        outputs = np.zeros((len(test_df),3*weeks_ahead))
        ###### Process Data (we want an array with current, last week, and two weeks ago cases)

        split_df = [group for _, group in test_df.groupby('Region')]

        

        for n,df in tqdm(enumerate(split_df)):
            
            region = np.unique(df['Region'])[0] 
            current_rhos = self.output_df[region + ' Rhos']
            sigmas = self.output_df[region + ' Sigmas']

            previous_previous_cases = self.cases[n,0]
            previous_cases = self.cases[n,1]
            test = df['Cases'].to_numpy().astype(int)
            test[test==0] = 1
            mapping = df['Ind'].to_numpy().astype(int)
            for week in tqdm(range(len(test))):
                if week > 0:
                    current_rhos = self.Update_Rho(current_rhos,previous_cases,previous_previous_cases)

                cases = predict_single(current_rhos,weeks_ahead,sigmas,previous_cases)

                medians = np.median(cases,0)
                l5 = np.quantile(cases,0.05)
                u5= np.quantile(cases,0.95)

                outputs[mapping[week],:weeks_ahead] = medians
                outputs[mapping[week],weeks_ahead:2*weeks_ahead] = l5
                outputs[mapping[week],2*weeks_ahead:] = u5
                previous_previous_cases = previous_cases+0
                previous_cases = test[week]
         
                
                
                
                
            





        ##### Trim to test data
        columns = []
        for ahead in range(weeks_ahead):
            test_df['Prediction ' + str(ahead+1)] = outputs[:,ahead]
            test_df['Prediction L5 ' + str(ahead+1)] = outputs[:,ahead + weeks_ahead]
            test_df['Prediction U5 ' + str(ahead+1)] = outputs[:,ahead + weeks_ahead*2]

        
        return test_df
    

    
    ################################################
    def name(self):
        
        return model.name
    