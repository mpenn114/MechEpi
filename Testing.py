import pandas as pd
import numpy as np

def Process_Scores(predictions,dfout_test,weeks_ahead,name):
    to_append = [name]
    dfout_test['Ind'] = np.arange(len(dfout_test))
    true_data = np.zeros((len(dfout_test),weeks_ahead))
    
    split_df = [group for _, group in dfout_test.groupby('Region')]
    curr_row = 0
    for df in split_df:
        for ahead in range(weeks_ahead):
            df['Forward '+ str(ahead+1)] = df['Cases'].shift(periods = -ahead-1)
        df = df.fillna(-1)
        true_data[df['Ind'].to_numpy().astype(int)] = df.to_numpy()[:,-weeks_ahead:]
            
            
            
        curr_row += len(df)
    
    
    pred_data = predictions.to_numpy()[:,-weeks_ahead:].astype(float)
    
    for week in range(weeks_ahead):
        
        to_append.append(np.sqrt(np.mean(np.square(true_data[true_data[:,week] !=-1 ,week] - pred_data[true_data[:,week] !=-1 ,week] ))))
        
    return to_append



def ILI_Data():
    df = pd.read_csv('ILINet.csv',skiprows=1)[['REGION','YEAR','WEEK','ILITOTAL']]
    df = df[df['ILITOTAL']!='X']
    df_out = df[['REGION','WEEK','ILITOTAL','YEAR']]
    df_out.columns = ['Region','Week','Cases','Year']
    
    
    dfout_train = df_out[df_out['Year'] + df_out['Week']*0.01 < 2015.25]
    
    
    dfout_test = df_out[(df_out['Year'] + df_out['Week']*0.01 >= 2015.25)&(df_out['Year'] + df_out['Week']*0.01 < 2019.25)]
    
    
    return dfout_train[['Region','Week','Cases']],dfout_test[['Region','Week','Cases']],df_out[['Region','Week','Cases']]



def ILI_Test(models,weeks_ahead):
    
    
    #########################################################################
    
    '''
    
    This is the only thing we'll need to change to run this on a different dataset.
    
    '''
    
    
    dfout_train,dfout_test,dfout = ILI_Data()
    
    ##########################################################################
    
    #Creating Output Array
    
    output_predictions = []
    
    sc_array = ['Model']
    for week in range(weeks_ahead):
        sc_array.append('Week ' + str(week + 1))
    
    scores = [sc_array]
    
    
    
    ############################################################################
    
    #Testing models
    
    for model in models:
        predictor = model()
        
        predictor.train(dfout_train)
        
        predictions = predictor.predict(dfout_test,dfout,weeks_ahead)

        score_row = Process_Scores(predictions,dfout_test,weeks_ahead,predictor.name)
        
        scores.append(score_row)
        
        
    ##############################################################################    
        
    scores = pd.DataFrame(scores[1:],columns = scores[0])

    
    return scores
        
        
        
        
    
    


    
