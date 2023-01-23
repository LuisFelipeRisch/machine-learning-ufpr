import constants
import pandas as pd

def sanitize_data(desired_day) -> pd.DataFrame:
    """
    Explanation: This function sanitize data from csv to filter columns.
    select columns that are below the required day, that do not have NaN in any of the features and that the target feature has a value above zero.

    @desired_day: The desired day to select the columns that are below it

    @return: The function returns DataFrame of pandas module
    """

    if desired_day == 1: return pd.DataFrame()

    sanitized_data_df = pd.read_csv(constants.DATA_PATH, header = None, sep=',')
    sanitized_data_df.columns = ["DATA", "DBZH", "DBZV", "KDP", "ZDR", "RHOHV", "TP_EST"]
    sanitized_data_df = sanitized_data_df[    sanitized_data_df["DBZH"].notnull() & 
                                        sanitized_data_df["DBZV"].notnull() & 
                                        sanitized_data_df["KDP"].notnull() &
                                        sanitized_data_df["ZDR"].notnull() &
                                        sanitized_data_df["RHOHV"].notnull() & 
                                        sanitized_data_df["TP_EST"].notnull() & 
                                        sanitized_data_df["TP_EST"] > 0.0 ]
    sanitized_data_df['DATA'] = pd.to_datetime(sanitized_data_df['DATA'], format='%Y-%m-%d %H:%M')
    sanitized_data_df = sanitized_data_df.loc[sanitized_data_df['DATA'] < f"2021-01-{str(desired_day)}"]
    sanitized_data_df['DAY'] = sanitized_data_df.apply(lambda x: x['DATA'].day, axis=1)

    return sanitized_data_df