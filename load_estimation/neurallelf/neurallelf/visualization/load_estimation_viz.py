''' 
A module for visualizing load estimation scenarios.
'''

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


ait_colors = {'bordeaux':'#79151d',
            'grau':'#7c8388',
            'türkis':'#31b7bc',
            'violet':'#470f51',
            '50grau':'lightgray',
            'schwarz':'#000c20'}


def plot_predict_voltage(nndto,buses=None,save=False,path='',minx=2000,maxx=2100):
    '''
    Plot the original trace with an overlay of the ANN and the linear regression predictions for a test dataset.
    Up to 13 traces are plotted. The desired busbars may be specified, otherwise the first up to 13 buses are taken.

    Args:
        nndto: a loaded nndto object
        dataset: a loaded dataset object
        buses (list): the original column names of busbars or None
    '''
    model = nndto.best_model[0]  # just picks the first model
    modelLR = nndto.model_linreg

    prediction = model.predict(nndto.Xtest.values)
    predictionLR = modelLR.predict(nndto.Xtest.values)
    ytest_data = nndto.ytest.values

    n_ax = len(ytest_data[0,:]) if len(ytest_data[0,:]) <= 13 else 13
    fig,axes = plt.subplots(n_ax,1,figsize=(5,1*n_ax+0.5),sharex=True)

    bus_names = buses if buses else nndto.ytest.columns[:n_ax]

    for idx,bus_name in enumerate(bus_names):
            scaler_y = nndto.scalers[1].loc[0,bus_name]
            ytest_rs = scaler_y.inverse_transform(ytest_data[:,idx].reshape(-1,1))
            predict_rs = scaler_y.inverse_transform(prediction[:,idx].reshape(-1,1))
            predictLR_rs = scaler_y.inverse_transform(predictionLR[:,idx].reshape(-1,1))

            r2 = r2_score(ytest_rs,predict_rs)
            r2_LR = r2_score(ytest_rs,predictLR_rs)

            axes[idx].plot(ytest_rs,label=f"Actual {bus_name[:-2]}",color=ait_colors['grau'],lw=2.5)
            axes[idx].plot(predictLR_rs,label=f"LR, R2: {r2_LR:.4f}",lw=1.8,color=ait_colors['schwarz'])
            axes[idx].plot(predict_rs,label=f"ANN, R2: {r2:.4f}",lw=1,color=ait_colors['bordeaux'])
            axes[idx].legend(loc=1,fontsize=5,framealpha=1)
            axes[idx].set_xlim(minx,maxx)
            axes[idx].set_ylim(min(ytest_rs[minx:maxx]),max(ytest_rs[minx:maxx]))
            axes[idx].set_ylabel("Voltage [pu]")
        
    axes[-1].set_xlabel("Record index test set")
    fig.suptitle(f"Predictions and acutal values for busbars")
    plt.tight_layout()
    if save:
        path.mkdir(exist_ok=True)
        plt.savefig(path / f"ANN LR pred.png",dpi=200 )

    
def plot_actual_pred_voltage(nndto,buses=None,save=False,path='',name='',index=0):
    '''
    Plot the residuals of the prediction for a test dataset.
    Up to 12 graphs are plotted. The desired busbars may be specified, otherwise the first up to 12 buses are taken.

    Args:
        nndto: a loaded nndto object
        dataset: a loaded dataset object
        buses (list): the original column names of busbars or None
    '''
    model = nndto.best_model[0]  # just picks the first model
    modelLR = nndto.model_linreg

    prediction = model.predict(nndto.Xtest.values)
    predictionLR = modelLR.predict(nndto.Xtest.values)
    ytest_data = nndto.ytest.values

    n_ax = len(ytest_data[0,:]) if len(ytest_data[0,:]) <= 12 else 12
    fig,axes = plt.subplots(n_ax//2,2, figsize=(7,1.5*n_ax//2+0.5))

    bus_names = buses if buses else nndto.ytest.columns[:n_ax]

    for idx,bus_name in enumerate(bus_names):
            scaler_y = nndto.scalers[1].loc[0,bus_name]
            ytest_rs = scaler_y.inverse_transform(ytest_data[:,idx].reshape(-1,1))
            predict_rs = scaler_y.inverse_transform(prediction[:,idx].reshape(-1,1))
            predictLR_rs = scaler_y.inverse_transform(predictionLR[:,idx].reshape(-1,1))

            axes[idx//2,idx%2].scatter(ytest_rs, predictLR_rs, label=f"LR - {bus_name[:-2]}", color=ait_colors['grau'], alpha=0.1, s=10)
            axes[idx//2,idx%2].scatter(ytest_rs, predict_rs, label="ANN", color=ait_colors['türkis'], alpha=0.1, s=10)
            axes[idx//2,idx%2].legend(loc=1,fontsize=5,framealpha=1)
            
            axes[idx//2,idx%2].set_ylabel("Pred. voltage [pu]")
        
    axes[idx//2,0].set_xlabel("Actual voltage [pu]")
    axes[idx//2,1].set_xlabel("Actual voltage [pu]")
    fig.suptitle(f"Predictions and acutal values for busbars")
    plt.tight_layout()
    if save:
        path.mkdir(exist_ok=True)
        plt.savefig(path / f"ANN LR actual vs prediction {name}_{index}.png",dpi=200 )


def plot_residuals_voltage(nndto,buses=None,save=False,path='',name='',index=0):
    '''
    Plot the residuals of the prediction for a test dataset.
    Up to 12 graphs are plotted. The desired busbars may be specified, otherwise the first up to 12 buses are taken.

    Args:
        nndto: a loaded nndto object
        dataset: a loaded dataset object
        buses (list): the original column names of busbars or None
    '''
    model = nndto.best_model[0]  # just picks the first model
    modelLR = nndto.model_linreg

    prediction = model.predict(nndto.Xtest.values)
    predictionLR = modelLR.predict(nndto.Xtest.values)
    ytest_data = nndto.ytest.values

    n_ax = len(ytest_data[0,:]) if len(ytest_data[0,:]) <= 12 else 12
    fig,axes = plt.subplots(n_ax//2,2,figsize=(6,1*n_ax//2+0.5),sharex=True)

    bus_names = buses if buses else nndto.ytest.columns[:n_ax]

    for idx,bus_name in enumerate(bus_names):
            scaler_y = nndto.scalers[1].loc[0,bus_name]
            ytest_rs = scaler_y.inverse_transform(ytest_data[:,idx].reshape(-1,1))
            predict_rs = scaler_y.inverse_transform(prediction[:,idx].reshape(-1,1))
            predictLR_rs = scaler_y.inverse_transform(predictionLR[:,idx].reshape(-1,1))

            residuals = predict_rs - ytest_rs
            residualsLR = predictLR_rs - ytest_rs

            axes[idx//2,idx%2].hist(residualsLR, label=f"LR - {bus_name[:-2]}", color=ait_colors['grau'], alpha=0.5)
            axes[idx//2,idx%2].hist(residuals, label="ANN", color=ait_colors['türkis'], alpha=0.5)
            axes[idx//2,idx%2].legend(loc=1,fontsize=5,framealpha=1)
            axes[idx//2,idx%2].set_ylabel("Frequency")
        
    axes[idx//2,0].set_xlabel("Residual voltage [pu]")
    axes[idx//2,1].set_xlabel("Residual voltage [pu]")
    fig.suptitle(f"Residual values for busbars")
    plt.tight_layout()
    if save:
        path.mkdir(exist_ok=True)
        plt.savefig(path / f"ANN LR residuals {name}_{index}.png",dpi=200 )



def get_x_labels(dataset,ledto):
    x_tick_labels_raw = [dataset.pq_df.columns[col_list].values for col_list in ledto.pq_ind_known]
    x_tick_labels = []
    for idx, sublist in enumerate(x_tick_labels_raw):
        if len(sublist)>10:
            x_tick_labels.append(len(sublist))
        else:
                x_tick_labels.append(sorted(set([(element[:-2]) for element in sublist])))
    return (x_tick_labels_raw, x_tick_labels)


def plot_metric_R2_scenarios(ledto,dataset,metric_name='',save=False,fig_path='',name='',metric_factor=1):
    '''Plot the R2 metric and a chosen metric for each scenario of a grid and for each unknown load.
    Args:
        ledto (LEDTO): a load estimation data transfer object with training results
        metric_factor: scaling factor e.g. to go from MW to kW
    '''
    fig,axes = plt.subplots(2,1,figsize=(6,5+len(ledto.pq_ind_known[-1])*0.08),sharex=True)
    for idx_sc,nndto in enumerate(ledto.nndto_list):
        for tup in nndto.metric_df.itertuples():
            # check that the column is from a load
            if (tup.ycolumn[-2:]=='_P') or (tup.ycolumn[-2:]=='_Q'):
                if tup.ycolumn[-1]=='P': col = ait_colors['bordeaux']
                else: col = ait_colors['50grau']
                axes[0].scatter(idx_sc,tup.r2,alpha=0.6,color=col) 
                axes[0].text(idx_sc+0.07,tup.r2-0.02,f"{tup.ycolumn}",fontsize=5)
                axes[1].scatter(idx_sc,tup.metric*metric_factor,alpha=0.6,color=col)
                axes[1].text(idx_sc+0.07,tup.metric*metric_factor,f"{tup.ycolumn}",fontsize=5)
    axes[1].set_xticks(range(len(ledto.pq_ind_known)))
    axes[1].set_xticklabels(get_x_labels(dataset,ledto)[1],rotation=90,fontsize=8)
    axes[0].set_ylabel("R2")
    axes[0].set_ylim(0,1.05)
    axes[1].set_xlabel("Scenario [known load ID or number]")
    axes[1].set_ylabel(f"{metric_name}")
    axes[0].set_title(f"{name}: R2 values of each P and Q load prediction for scenarios")
    axes[1].set_title(f"{metric_name} for each P and Q load prediction for scenarios")
    plt.tight_layout()
    if save:
        fig_path.mkdir(exist_ok=True)
        plt.savefig(fig_path / f"LE scenorios metric R2 {name}.png",dpi=300 )


def plot_metric_R2_scenarios_buses(ledto,dataset,exlude_buses_in_evaluation,metric_name='',save=False,fig_path='',name='',metric_factor=1):
    '''Plot the R2 metric and a chosen metric for each scenario of a grid and for each unknown bus.
    Args:
        ledto (LEDTO): a load estimation data transfer object with training results
        metric_factor: scaling factor 
    '''
    col = ait_colors['türkis']
    fig,axes = plt.subplots(2,1,figsize=(6,5+len(ledto.pq_ind_known[-1])*0.08),sharex=True)
    for idx_sc,nndto in enumerate(ledto.nndto_list):
        for tup in nndto.metric_df.itertuples():
            # check that the column is from a load
            if (tup.ycolumn[-2:]=='_V') and (tup.ycolumn not in exlude_buses_in_evaluation):
                axes[0].scatter(idx_sc,tup.r2,alpha=0.7,color=col) 
                axes[0].text(idx_sc+0.07,tup.r2,f"{tup.ycolumn[:-2]}",fontsize=5)
                axes[1].scatter(idx_sc,tup.metric*metric_factor,alpha=0.7,color=col)
                axes[1].text(idx_sc+0.07,tup.metric*metric_factor,f"{tup.ycolumn[:-2]}",fontsize=5)
    axes[1].set_xticks(range(len(ledto.pq_ind_known)))
    axes[1].set_xticklabels(get_x_labels(dataset,ledto)[1],rotation=90,fontsize=7)
    axes[0].set_ylabel("R2")
    #axes[0].set_ylim(0.8,1.05)
    axes[1].set_xlabel("Scenario [known load/busbar ID or number]")
    axes[1].set_ylabel(f"{metric_name}")
    axes[0].set_title(f"{name}: R2 values of each busbar prediction for scenarios")
    axes[1].set_title(f"{metric_name} for each busbar prediction for scenarios")
    plt.tight_layout()
    if save:
        fig_path.mkdir(exist_ok=True)
        plt.savefig(fig_path / f"LE scenorios metric R2 bus {name}.png",dpi=300 )


def plot_metric_R2_fraction(ledto,metric_name='MAE',save=False,fig_path='',name='',metric_factor=1):
    '''
    Plots averaged R2 / metric for each fraction of unknown loads in the scenarios for a grid.
    An unknown load is a single P/Q pair (length of pq_ind_unknown/2).
    Args:
        ledto (LEDTO): a load estimation data transfer object with training results and a filled metrics_summary_df dataframe
        metric_factor: scaling factor e.g. to go from MW to kW
    '''
    fig,axes = plt.subplots(2,1,figsize=(6,5),sharex=True)
    # P
    axes[0].scatter(ledto.metrics_summary_df['fraction'],ledto.metrics_summary_df['r2_avg_P'],alpha=0.3,color=ait_colors['bordeaux'],label='P')
    axes[1].scatter(ledto.metrics_summary_df['fraction'],ledto.metrics_summary_df['metric_avg_P']*metric_factor,alpha=0.3,color=ait_colors['bordeaux'],label='P')
    # Q
    axes[0].scatter(ledto.metrics_summary_df['fraction'],ledto.metrics_summary_df['r2_avg_Q'],alpha=0.3,color=ait_colors['grau'],label='Q')
    axes[1].scatter(ledto.metrics_summary_df['fraction'],ledto.metrics_summary_df['metric_avg_Q']*metric_factor,alpha=0.3,color=ait_colors['grau'],label='Q')
    
    axes[1].set_xticks(ledto.metrics_summary_df['fraction'].unique())
    axes[0].set_ylabel("R2")
    axes[0].set_ylim(0,1.05)
    axes[1].set_xlabel("Fraction of known loads")
    axes[1].set_ylabel(f"Metric {metric_name}")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle(f"Prediction R2 and {metric_name} vs. fraction of known loads {name}")
    plt.tight_layout()
    if save:
        fig_path.mkdir(exist_ok=True)
        plt.savefig(fig_path / f"ann metric R2 fraction {name}.png",dpi=200 )


def plot_metric_R2_fraction_bus(ledto,metric_name='MAE',save=False,fig_path='',name='',metric_factor=1):
    '''
    Plots averaged R2 / metric for buses for each fraction of unknown loads in the scenarios for a grid.
    An unknown load is a single P/Q pair (length of pq_ind_unknown/2).
    Args:
        ledto (LEDTO): a load estimation data transfer object with training results and a filled metrics_summary_df dataframe
        metric_factor: scaling factor e.g. to go from MW to kW
    '''
    fig,axes = plt.subplots(2,1,figsize=(6,5),sharex=True)
    # P
    # try:
    #     for tup in ledto.metrics_summary_df[['fraction','r2_avg_bus','r2_var_avg_bus']].itertuples():
    #         axes[0].vlines(tup.fraction,tup.r2_avg_bus-tup.r2_var_avg_bus/2,tup.r2_avg_bus+tup.r2_var_avg_bus/2)
    # except: pass
    axes[0].scatter(ledto.metrics_summary_df['fraction'],ledto.metrics_summary_df['r2_avg_bus'],alpha=0.4,color=ait_colors['türkis'],label='Busbar')
    axes[1].scatter(ledto.metrics_summary_df['fraction'],ledto.metrics_summary_df['metric_avg_bus']*metric_factor,alpha=0.3,color=ait_colors['türkis'],label='Busbar')
    
    axes[1].set_xticks(ledto.metrics_summary_df['fraction'].unique().round(4))
    axes[0].set_ylabel("R2")
    #axes[0].set_ylim(0,1.05)
    axes[1].set_xlabel("Fraction of known busbars")
    axes[1].set_ylabel(f"Metric {metric_name}")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle(f"Pred. R2 and {metric_name} vs. fraction of known busbars {name}")
    plt.tight_layout()
    if save:
        fig_path.mkdir(exist_ok=True)
        plt.savefig(fig_path / f"ann metric R2 fraction bus {name}.png",dpi=200 )


def plot_metric_R2_fraction_LR(ledto,metric_name='MAE',save=False,fig_path='',name='',metric_factor=1):
    '''
    Plots averaged R2 / metric for each fraction of unknown loads in the scenarios for a grid.
    An unknown load is a single P/Q pair (length of pq_ind_unknown/2).
    Args:
        ledto (LEDTO): a load estimation data transfer object with training results and a filled metrics_summary_df dataframe
        metric_factor: scaling factor e.g. to go from MW to kW
    '''
    fig,axes = plt.subplots(2,1,figsize=(6,5),sharex=True)
    # P
    axes[0].scatter(ledto.metrics_summary_df['fraction'],ledto.metrics_summary_df['r2_avg_P'],alpha=0.4,color=ait_colors['bordeaux'],label='P')
    axes[0].scatter(ledto.metrics_summary_df['fraction'],ledto.metrics_summary_df['r2_avg_P_LR'],alpha=0.3,color=ait_colors['grau'],label='P - LR')
    axes[1].scatter(ledto.metrics_summary_df['fraction'],ledto.metrics_summary_df['metric_avg_P']*metric_factor,alpha=0.4,color=ait_colors['bordeaux'],label='P')
    axes[1].scatter(ledto.metrics_summary_df['fraction'],ledto.metrics_summary_df['metric_avg_P_LR']*metric_factor,alpha=0.3,color=ait_colors['grau'],label='P - LR')
    # Q
    #axes[0].scatter(ledto.metrics_summary_df['fraction'],ledto.metrics_summary_df['r2_avg_Q'],alpha=0.3,color='tab:green',label='Q')
    #axes[1].scatter(ledto.metrics_summary_df['fraction'],ledto.metrics_summary_df['metric_avg_Q']*metric_factor,alpha=0.3,color='tab:green',label='Q')
    
    axes[1].set_xticks(ledto.metrics_summary_df['fraction'].unique())
    axes[0].set_ylabel("R2")
    axes[0].set_ylim(0,1.05)
    axes[1].set_xlabel("Fraction of known loads")
    axes[1].set_ylabel(f"Metric {metric_name}")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle(f"Prediction R2 and {metric_name} vs. fraction of known loads {name}")
    plt.tight_layout()
    if save:
        fig_path.mkdir(exist_ok=True)
        plt.savefig(fig_path / f"ann linreg metric R2 fraction {name}.png",dpi=200 )


def plot_metric_R2_fraction_general(metrics_summary_df,ycolsR2,ycolMet,ycolors,ylabels,title,metric_name='MAE',save=False,fig_path='',name='',metric_factor=1):
    '''
    Plots averaged R2 / metric for each fraction of unknown loads in the scenarios for a grid.
    An unknown load is a single P/Q pair (length of pq_ind_unknown/2).
    Args:
        metrics_summary_df:  metrics_summary_df dataframe from evaluation
        metric_factor: scaling factor e.g. to go from MW to kW
    '''
    fig,axes = plt.subplots(2,1,figsize=(6,5),sharex=True)
    
    for idx,ycolR2 in enumerate(ycolsR2):
        axes[0].scatter(metrics_summary_df['fraction'],metrics_summary_df[ycolR2],alpha=0.35,color=ycolors[idx],label=ylabels[idx])
        axes[1].scatter(metrics_summary_df['fraction'],metrics_summary_df[ycolMet[idx]]*metric_factor,alpha=0.35,color=ycolors[idx],label=ylabels[idx])
    
    axes[1].set_xticks(metrics_summary_df['fraction'].unique().round(3))
    axes[0].set_ylabel("R2")
    #axes[0].set_ylim(0,1.05)
    axes[1].set_xlabel("Fraction of known loads/busbars")
    axes[1].set_ylabel(f"Metric {metric_name}")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle(f"{title} R2 and {metric_name} vs. fraction of known loads/busbars {name}")
    plt.tight_layout()
    if save:
        fig_path.mkdir(exist_ok=True)
        plt.savefig(fig_path / f"{title} metric R2 fraction {name}.png",dpi=200 )