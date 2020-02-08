###
###  Solution for test case (Data Scientist) from Jooble
###  by Taras Lazariev, 2020
###

import numpy as np
import pandas as pd

def get_job_data(file_name):
    """
    Read tab separated file with id_job and jobs features combined (',' concatenation) in feature sets,
    process data for later manipulation and analysis
        
    Parameters
    ----------
    file_name : str
        tab separated file with arbitrary header and feature sets
        
    Returns
    -------
    output_df: DataFrame
        DataFrame with id_job add features in separate columns for each feature from each features sets:
            id_job feature_2_0 feature_2_1 ... feature_n_n.
        NaN is used as empty data filler
        
    feature_set_patterns: list
        List of feature sets names
        [feature_2, feature_3 ...]
    """
    
    input_df = pd.read_csv(file_name, sep='\t', skiprows = 1,  header = None)

    output_df = pd.DataFrame(columns = ['id_job','feature_set', 'features'])

    # loop through feature sets and split feture_set_id and features
    for column in input_df.columns[1:]:
        temp = pd.concat(
            [
                input_df.iloc[:,0].to_frame(),
                input_df.iloc[:,column].str.split(',', 1, expand=True)
            ],
            axis = 1)
        temp.columns = ['id_job','feature_set', 'features']

        output_df = pd.concat([output_df, temp], axis = 0)

    feature_set_patterns = []

    # group by feature_set, split features and create new columns with proper name (example: feature_2_0) 
    grouped  = output_df.groupby('feature_set')
    for set_id, group in grouped:

        feature_set_name = 'feature' + '_' + set_id  
        feature_set_patterns.append(feature_set_name)

        output_df = output_df.join(
            group['features'].str.split(',', -1, expand=True)
            .astype('int')
            .add_prefix(feature_set_name + '_')
            )

 
    output_df.drop(columns=['features', 'feature_set'], inplace = True)
    # remove duplicates (duplicates are appeared when there are more then one feature set)
    output_df = output_df[~output_df.index.duplicated(keep = 'first')]
    
    return output_df, feature_set_patterns
    
    

def creat_feature_metric_names(df, feature_set_pattern, metric_name):
    """
    Create list of metric names from list of features
    
    Parameters
    ----------
    df : DataFrame
    
    feature_set_pattern: str
         pattern of feature set:
         example: 'feature_2'

    metric_name: str
         metric name
         example: 'stand'
     
    Returns
    -------
    feature_metric_names: list
         list of metric names:
             [feature_2_stand_0, feature_2_stand_1, ... feature_2_n]
    """

    fetures = df.columns.str.contains(feature_set_pattern)
    
    feature_names = list(df.columns[fetures])
    
    feature_metric_names = [ 
                x.replace(
                    feature_set_pattern ,
                    feature_set_pattern + '_' + metric_name
                    )
                    for x in feature_names
                ]

    return feature_metric_names




def calc_z_score(df, feature_set_pattern, features_mean, features_std, metric_name_suffix = 'stand'):
    """
    Create new DataFrame with Z-score for features that match feature_set_pattern
    Z-score = 
        (fature_value - features_mean) / features_std 
     
    Parameters
    ----------
    df : DataFrame
     
    feature_set_pattern: str
        pattern of feature set:
        example: 'feature_2'

    features_mean: Series
        mean values for feature in feature set 
        example manual: pd.Series(values, index = [feature_2_0, feature_2_1 ....])
        example from train data:
             
         >>> train, train_feature_set_patterns = get_job_data('train.tsv
         >>> train_features_names = train.columns.str.contains('feature_2')
         >>> train_mean = train.loc[:,train_features_names].mean()
     
    features_std: Series
        std values for feature in feature set 
        example manual: pd.Series(values, index = [feature_2_0, feature_2_1 ....])
        example from train data:
             
        >>> train, train_feature_set_patterns = get_job_data('train.tsv
        >>> train_features_names = train.columns.str.contains('feature_2')
        >>> train_mean = train.loc[:,train_features_names].std()
     
    metric_name_suffix: str, default 'stand'
        prefix to be add in column names
        example: feature_2_stand_0, feature_2_stand_1 ...
     
    Returns
    -------
    fetures_z_score: DataFrame
        DataFrame with Z-score for features that match feature_set_pattern
    """
    
    features = df.columns.str.contains(feature_set_pattern)

    fetures_z_score = (df.loc[:,features] - features_mean) / features_std 

    fetures_z_score.columns = creat_feature_metric_names(df, feature_set_pattern, metric_name_suffix)

    return fetures_z_score



def calc_max_feture_index(df, feature_set_pattern):
    """
    get index of max feature for features that match feature_set_pattern
     
    Parameters
    ----------
    df : DataFrame
    
    feature_set_pattern: str
        pattern of feature set:
        example: 'feature_2'
         
    Returns
    -------
    max_feature_ind: Series
        Series with indexes of max feature in feature set
    """

    features = df.columns.str.contains(feature_set_pattern)

    data = df.loc[:,features]

    max_feature_name = data.idxmax(axis = 'columns')
    max_feature_ind = max_feature_name[max_feature_name.notna()].apply(lambda x: int(x.split('_')[-1]))
    max_feature_ind.name = 'max_' + feature_set_pattern + '_index'

    return max_feature_ind 
    

def calc_max_feture_abs_mean_diff(df, feature_set_pattern):
    """
    get abs_mean_diff of max feature for features that match feature_set_pattern
    abs_mean_diff = 
        abs(
            value(max_feature_2_ind) - mean(max_feature_2_ind)
            )
    Parameters
    ----------
    df : DataFrame
     
    feature_set_pattern: str
        pattern of feature set:
        example: 'feature_2'
         
    Returns
    -------
    max_feature_abs_mean_diff: Series
        Series with absolute deviation of max feature from this feature mean
    """

    features = df.columns.str.contains(feature_set_pattern)

    data = df.loc[:,features]

    max_feature_name = data.idxmax(axis = 'columns')
    max_feature_value = data.max(axis = 'columns')
    all_feature_mean = data.mean(axis = 0).to_dict()
    max_feature_mean = max_feature_name[max_feature_name.notna()].apply(lambda x: all_feature_mean[x])


    max_feature_abs_mean_diff = (max_feature_value - max_feature_mean).abs()
    max_feature_abs_mean_diff.name = 'max_' + feature_set_pattern + '_abs_mean_diff'

    return max_feature_abs_mean_diff 


######################################################################################
###
###  Example solution for test case
###
######################################################################################

# read tran dat
train, train_feature_set_patterns = get_job_data('train.tsv')
# read test data
test, test_feature_set_patterns = get_job_data('test.tsv')

# define feature set patter
feature_set_pattern = 'feature_2'
# calculate mean adn std from train data
train_features_names = train.columns.str.contains(feature_set_pattern)

train_mean = train.loc[:,train_features_names].mean()
train_std = train.loc[:,train_features_names].std()

# create outpt DataFrame
output = pd.concat(
        (
            test['id_job'],
            calc_z_score(
                test, 
                feature_set_pattern,
                features_mean = train_mean,
                features_std = train_std,
                metric_name_suffix = 'stand'),
            calc_max_feture_index(test, feature_set_pattern),
            calc_max_feture_abs_mean_diff(test, feature_set_pattern)
        ), axis = 1
    )

# write output data
output.to_csv('test_proc.tsv',sep = '\t', index=False, na_rep='NA')




'''

######################################################################################
###
###  Example solution for more general test case
###
######################################################################################

# read tran dat
train, train_feature_set_patterns = get_job_data('train.add.tsv')
# read test data
test, test_feature_set_patterns = get_job_data('test.add.tsv')

# create output DataFrame
output = test['id_job']

# loop through feature sets
for train_pat, test_pat in zip(train_feature_set_patterns, test_feature_set_patterns): 
    train_features_names = train.columns.str.contains(train_pat)

    train_mean = train.loc[:,train_features_names].mean()
    train_std = train.loc[:,train_features_names].std()

    output = pd.concat(
        (
            output,
            calc_z_score(
                test, 
                test_pat,
                features_mean = train_mean,
                features_std = train_std,
                metric_name_suffix = 'stand'),
            calc_max_feture_index(test, test_pat),
            calc_max_feture_abs_mean_diff(test, test_pat)
        ), axis = 1
    )

# write output data
output.to_csv('test.add_proc.tsv',sep = '\t', index=False, na_rep='NA')

'''