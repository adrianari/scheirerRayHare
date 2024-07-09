from scipy import stats


def scheierRayHare(factor1, factor2, goal_var, df):
    """ 
    Scheier Ray Hare (after https://github.com/jpinzonc/Scheirer-Ray-Hare-Test/blob/master/Scheirer-Ray-Hare%20Test.ipynb)

    Input:

    factor1:    Column name in df which represents Factor 1
    factor2:    Column name in df which represents Factor 2
    goal_var:   Column name in df which represents Measurement
    df:         Dataframe

    """

    df['rank'] = df[goal_var].sort_values().rank(numeric_only = float)

    rows = df.groupby([factor1], as_index = False).agg({'rank':['count', 'mean', 'var']}).rename(columns={'rank':'row'})
    rows.columns = ['_'.join(col) for col in rows.columns]
    rows['row_mean_rows'] = rows.row_mean.mean()
    rows['sqdev'] = (rows.row_mean - rows.row_mean_rows)**2


    cols = df.groupby([factor2], as_index = False).agg({'rank':['count', 'mean', 'var']}).rename(columns={'rank':'col'})
    cols.columns = ['_'.join(col) for col in cols.columns]
    cols['col_mean_cols'] = cols.col_mean.mean()
    cols['sqdev'] = (cols.col_mean-cols.col_mean_cols)**2


    data_sum         = df.groupby([factor1, factor2], as_index = False).agg({'rank':['count', 'mean', 'var']}).round(1)
    data_sum.columns = ['_'.join(col) for col in data_sum.columns]

    nobs_row   = rows.row_count.mean()
    nobs_total = rows.row_count.sum()
    nobs_col   = cols.col_count.mean()

    Columns_SS = cols.sqdev.sum()*nobs_col
    Rows_SS    = rows.sqdev.sum()*nobs_row
    Within_SS  = data_sum.rank_var.sum()*(data_sum.rank_count.min()-1)
    MS         = df['rank'].var()
    TOTAL_SS   = MS * (nobs_total-1)
    Inter_SS   = TOTAL_SS - Within_SS - Rows_SS - Columns_SS

    H_rows = Rows_SS/MS
    H_cols = Columns_SS/MS
    H_int  = Inter_SS/MS

    df_rows   = len(rows)-1
    df_cols   = len(cols)-1
    df_int    = df_rows*df_cols
    df_total  = len(df)-1
    df_within = df_total - df_int - df_cols - df_rows

    p_rows  = 1-stats.chi2.cdf(H_rows, df_rows)
    p_cols  = round(1-stats.chi2.cdf(H_cols, df_cols),4)
    p_inter = round(1-stats.chi2.cdf(H_int, df_int),4)

    return (p_rows, p_cols, p_inter)