import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

def calculateGeneExpression(counts, my_annotation):
    isec = my_annotation.set_index("ensembl_gene_id").sort_index(ascending=True).index.intersection(counts.index)
    anot = my_annotation.set_index("ensembl_gene_id").sort_index(ascending=True)
    anot = anot[~anot.index.duplicated(keep='first')]
    
    leng = pd.read_csv("Genes_length_Homo_Sapiens.csv").set_index("Gene_stable_ID")
    all_metadata = anot.merge(leng, right_index=True, left_index=True)
    ready = counts.loc[all_metadata.index]
    
    #Normalization
    column_sums = np.sum(ready, axis=0)
    sizeFactors = np.asarray(column_sums)
    log_counts = np.log2(ready + 1)
    log_sizeFactors = np.log2(sizeFactors / (10 ** 6))
    y = log_counts.apply(lambda row: row - log_sizeFactors, axis=1)
    
    subindex = np.where(np.mean(ready, axis=1) > 50)[0]
    gc_content = all_metadata.percentage_gene_gc_content.to_numpy()
    length = all_metadata.Gene_length.to_numpy()
    x1 = fix_predictor(gc_content, subindex)
    x2 = fix_predictor(np.log2(length/1000), subindex)
    
    df_fit = pd.DataFrame()
    df_fit = df_fit.assign(x1_fit=x1["fit"], x2_fit=x2["fit"])
    df_out = pd.DataFrame()
    df_out = df_out.assign(x1_out=x1["out"], x2_out=x2["out"])
    
    df1 = pd.DataFrame({'x1': x1["grid"], 'x2': [pd.Series(x2["grid"]).median()] * len(x1["grid"])})
    df2 = pd.DataFrame({'x1': [pd.Series(x1["grid"]).median()] * len(x2["grid"]), 'x2': x2["grid"]})
    df_res = pd.concat([df1, df2])
    df_func = pd.DataFrame()
    df_func = df_func.assign(x1=x1["grid"])
    x1_knots = x1["knots"]
    x2_knots = x2["knots"]
    
    regr = fit_quantile_regression(y.iloc[subindex], df_fit, df_out, df_res, df_func, x1_knots, x2_knots, tau=0.5)
    fitted = np.column_stack([res['fitted'] for res in regr])
    func = np.column_stack([res['func'] for res in regr])
    
    k = np.argsort(gc_content[subindex])[len(subindex) // 2]
    offset0 = np.median(fitted[subindex[k]])
    
    residuals = y - fitted
    offset = offset0 + residuals - y
    cqnValues = y+offset
    expressionMatrix = cqnValues - cqnValues.min().min() + 1
    
    expressionMatrix = expressionMatrix[~expressionMatrix.index.duplicated(keep=False)] #Chequear...keep
    expressionMatrix.index = anot.loc[expressionMatrix.index].external_gene_name
    expressionMatrix = expressionMatrix[~expressionMatrix.index.duplicated(keep=False)] #Creo que tendría sentido first o last
    
    return expressionMatrix



def fix_predictor(zz, subindex, varname="B"):
    zz_fit = zz[subindex]
    knots = np.quantile(zz_fit, q=[0.025, 0.25, 0.50, 0.75, 0.975]) + np.array([0.01, 0, 0, 0, -0.01])
    grid = np.linspace(start=min(zz_fit), stop=max(zz_fit), num=101)
    zz_out = zz.copy()
    zz2 = zz[~subindex]
    zz2[zz2 < min(zz_fit)] = min(zz_fit)
    zz2[zz2 > max(zz_fit)] = max(zz_fit)
    zz_out[~subindex] = zz2
    return {'fit': zz_fit, 'out': zz_out, 'knots': knots, 'grid': grid}


def fit_quantile_regression(yfit, df_fit, df_out, df_res, df_func, x1_knots=None, x2_knots=None, tau=0.5):
    regr = []
    for ii in range(yfit.shape[1]):
        print(ii)
        auxiliar = dict(y =yfit.iloc[:, ii], x1 = df_fit.x1_fit.to_numpy(), x2=df_fit.x2_fit.to_numpy(), knots1=x1_knots, knots2=x2_knots)
        mod = smf.quantreg('y ~ cr(x1, knots=knots1) + cr(x2, knots=knots2)', auxiliar)
        fit = mod.fit(q=tau, max_iter=750, p_tol=0.0001)
        aux2 = dict(x1 = df_out.x1_out.to_numpy(), x2=df_out.x2_out.to_numpy(), knots1=x1_knots, knots2=x2_knots)
        aux3= dict(x1 = df_res.x1.to_numpy(), x2=df_res.x2.to_numpy(), knots1=x1_knots, knots2=x2_knots)
        
        fitted = fit.predict(aux2, transform=True)
        func = fit.predict(aux3, transform=True)
        regr.append({'fitted': fitted, 'func': func, 'coef': fit.params})
    return regr


