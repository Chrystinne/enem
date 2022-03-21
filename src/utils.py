import pandas as pd
import dask.dataframe as dd
import json


def load_data_year(year, cols):
    file = f'../../../microdados_anos/MICRODADOS_ENEM_{year}.csv'

    try:
        df = dd.read_csv(file, encoding='cp1252', sep=';', usecols=cols, assume_missing=True)
    except:
        try:
            df = dd.read_csv(file, encoding='cp1252', usecols=cols, assume_missing=True)
        except:
            df = dd.read_csv(file, encoding='cp1252', sep=';', assume_missing=True)
            df = df.rename(columns={'NO_MUNICIPIO_PROVA': 'NO_MUNICIPIO_RESIDENCIA', 
                                        'SG_UF_PROVA': 'SG_UF_RESIDENCIA', 'TP_FAIXA_ETARIA': 'NU_IDADE'})[cols]
    
    return df

def load_parquets(path, n_parts=None):
    df = dd.read_parquet(path)
    
    for p in range(1, n_parts):
        path.replace(f'{p-1}', str(p))
        df = dd.concat([df, dd.read_parquet(path)])
        
    return df

def union_datas(years, cols):
    df = load_data_year(years[0], cols)
    
    for y in years[1:]:
        df = dd.concat([df, load_data_year(y, cols)])
    return df

def info_sum_isna(df):
    return pd.DataFrame({'types': df.dtypes, 'missing': df.isna().compute().sum()})


def add_ses_income(x):
    income_dict = {'A': 0, 'B': 1, 'C': 1.5, 'D': 2, 'E': 2.5, 'F': 3, 'G': 4, 'H': 5, 'I': 6, 
                   'J': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 12, 'O': 15, 'P': 20}

    if(x['Q006'] == 'Q'): 
        x['TP_SES_INCOME'] = 'A'
    else: 
        per_capita = income_dict[x['Q006']]/x['Q005']
        
        if (per_capita <= 1): x['TP_SES_INCOME'] = 'E'
        elif (per_capita <= 4):  x['TP_SES_INCOME'] = 'D'
        elif (per_capita <= 10): x['TP_SES_INCOME'] = 'C'
        else: x['TP_SES_INCOME'] = 'B'
        
    return x

def add_ses_points(x):
    with open("../datasets/sistema_pontos.json", encoding='utf-8') as json_:
        sistema_pontos = json.load(json_)
    
    points = 0 
    
    for index in x.index[25:]:
        if(index in sistema_pontos):
            if(index in ['Q001', 'Q002']):
                if(index == 'Q001'):
                    pm = sistema_pontos['Q001'][x['Q001']]
                    pp = sistema_pontos['Q002'][x['Q002']]

                    points += max(pm, pp)
                else:
                    points += 0 
            else:
                points += sistema_pontos[index][x[index]]
    
    if (points <= 16): x['TP_SES_POINTS'] = 'DE'
    elif (points <= 22):  x['TP_SES_POINTS'] = 'C2'
    elif (points <= 28): x['TP_SES_POINTS'] = 'C1'
    elif (points <= 37): x['TP_SES_POINTS'] = 'B2'
    elif (points <= 44): x['TP_SES_POINTS'] = 'B1'
    else: x['TP_SES_POINTS'] = 'A'
    
    return x

def add_null_values(x, df_groups):
    df_groups = df_final.groupby(['TP_ESCOLA', 'TP_ST_CONCLUSAO', 'SG_UF_RESIDENCIA']).mean()
    df_groups = df_groups.rename_axis(['TP_ESCOLA', 'TP_ST_CONCLUSAO', 'SG_UF_RESIDENCIA']).reset_index()
    
    if(sum(pd.isna(x)) > 0):
        df_x = df_groups.query('TP_ESCOLA==@x.TP_ESCOLA & TP_ST_CONCLUSAO==@x.TP_ST_CONCLUSAO & SG_UF_RESIDENCIA==@x.SG_UF_RESIDENCIA')
    
        for c in df_x.columns:
            if(pd.isna(x[c]) > 0):
                x[c] = df_x[c].values[0]
    return x