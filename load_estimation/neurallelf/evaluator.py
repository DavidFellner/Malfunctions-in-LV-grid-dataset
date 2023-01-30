''' 
Evaluate trained artificial neural networks in respect to linear regression models and visualize results.
'''
import os
from pathlib import Path
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

from plotting import plot_estimate_vs_target_by_load

from neurallelf.data.dataset_generators import DatasetGasen, DatasetTestGrids
from neurallelf.features.feature import *
from neurallelf.models.load_flow import *
from neurallelf.models.load_estimation import *
from neurallelf.visualization.load_estimation_viz import *

# set working directory 
wdir = Path(os.getcwd()) / Path(__file__)
wdir = wdir.parents[1]
#os.chdir(wdir)
print(os.getcwd())

ait_colors = {'bordeaux':'#79151d',
            'grau':'#7c8388',
            'türkis':'#31b7bc',
            'violet':'#470f51',
            '50grau':'lightgray',
            'schwarz':'#000c20'}

def eval(dir_results=None, setup=None, X=None, pad_factor=240, config=None):

    ### Parameter specifications
    ###################################

    from datetime import date
    # specify the directory for the run results:
    dir = f"ERIGrid_{dir_results.split('_')[-1]}"
    dir_results = f"ERIGrid_{dir_results.split('_')[-1]}_Setup_{setup}"
    if dir_results is None:
        # specify the directory for the run results:
        dir = "ERIGrid_phase1_Setup_A"
        dir_results = "ERIGrid_phase1"
    mode = 'LE'                                 # either of 'LE','DLEF','DLF','BusLF','LF' or 'LFfromLE'
    date = date.today()
    le_model_dir = f"{date.year}_{date.month}_{date.day}_LE_base"  # required only for mode 'LFfromLE'

    # specify datasets
    name_list = ['training_data']#['01','02','03','04','05','06']
    dir_name = f'raw_{dir}_'
    name_single = os.path.join(os.getcwd(),
                              f'raw_data\\{dir}_training_data\\PNDC_{dir}_training\\Load_estimation_training_data\\load_estimation_training_data_setup_{setup}.csv')  # use this in case of a single datafile, else put None

    os.path.isfile(name_single)

    # depending on the Dataset to use specify the graph data:
    #graph_pattern = ['CLUE_Test_','_Load_Bus_Mapping.txt']
    graph_name = os.path.join(os.getcwd(),
                              f'raw_data\\{dir}_training_data\\PNDC_{dir}\\Load_estimation_training_data\\{dir}_Load_Bus_Mapping.txt')

    # optional: depending on the Dataset to use
    #file_pattern = {'loads':'full_df_load.csv','voltages':'full_df_voltage.csv','pflow':'full_df_Pflow.csv', 'qflow':'full_df_Qflow.csv'}
    #file_pattern = {'loads':'narrow_df_load.csv','voltages':'narrow_df_voltage.csv','pflow':'narrow_df_Pflow.csv', 'qflow':'narrow_df_Qflow.csv'}
    #file_pattern = {'loads':'df_load.csv','voltages':'df_voltage.csv','pflow':'df_Pflow.csv', 'qflow':'df_Qflow.csv'}

    # specify always 'known' buses by original dataset name
    # (known includes: voltage, pflow and qflow at that bus)
    if dir_results.split('_')[1] == 'phase1':
        trafo_point = 'F2'  # pv use case
        known_buses = ['Test Bay ' + trafo_point, "Test Bay B1", "Test Bay F1"]  # [trafo_point,'node_57']
    else:
        trafo_point = 'B2'  # dsm use case
        known_buses = ['Test Bay ' + trafo_point, "Test Bay B1", "Test Bay A1",
                       "Test Bay C1"]  # [trafo_point,'node_57']                    #e.g. 'node01' these are not included in mode 'LF'

    exlude_buses_in_evaluation = []       # these buses are not considered for metrics evaluations e.g. node_144_V

    # specify parameters for evaluation run
    pred_plot_buses = None    # either list of busbar names or None (default selection)
    save_fig_scen = True   # generate a scenario plot for each grid
    save_fig_frac = True   # generate a fraction plot for each grid

    metric_rmse = True     # use RMSE instead of MSE

    # specify important paths
    data_path = Path("data")
    results_path = Path("model summaries")
    run_path = results_path / dir_results



    ### Raw data loading
    ###################################

    # load the data into dataset objects stored in dataset_dict
    dataset_dict = {}
    for name_str in name_list:
        directory = dir_name + name_str
        dataset = DatasetGasen()
        dataset.create_dataset(data_path,directory,name_single,graph_name)
        #dataset = DatasetTestGrids()
        #dataset.create_dataset(data_path,directory,name_str,file_pattern,graph_pattern)
        dataset.known_buses = [known_bus+'_V' for known_bus in known_buses]
        dataset_dict[name_str] = dataset


        #### my input
        if X == 'from_file':
            dataset_eval = DatasetGasen()
            file_path_input = os.path.join(os.getcwd(),
                                  f'raw_data\\{dir}_estimation_input_data\\Setup_{setup}\\Load_estimation_input_data\\load_estimation_input_data_setup_{setup}.csv')  # use this in case of a single datafile, else put None

            dataset_eval.create_dataset(data_path, file_path_input, '', graph_name)
        else:
            dataset_eval = None

        """merged_df = dataset_eval.df.copy()
        merged_df = merged_df.append(dataset.df, ignore_index=True)
        merged_df.to_csv('load_estimation_training_data_setup_A.csv')#, index=False)"""

        ### Load trained models
        ###################################

        # load results from scenarios in LEDTOs
        results_dict = {}
        for name in name_list:
            ledto = load_ledto(run_path,name,dataset_dict[name],load_models_too=True)
            results_dict[name] = create_ledto(ledto.pq_ind_known,dataset_dict[name].graph_df,dataset_dict[name].pq_df,dataset_dict[name].v_df,ledto_prefilled=ledto)


    ### Model evaluation
    ###################################

    known_buses_ind = [dataset.v_df.columns.get_loc(known_element) for known_element in dataset.known_buses]

    for name,ledto in results_dict.items():
        print(f"Grid: {name}")
        dataset = dataset_dict[name]

        # retrieve Xtest, ytest, scalers, df_avg and best_model
        for idx,scenario in enumerate(ledto.pq_ind_known):
            print (f'\tScenario: {scenario}')
            pq_known = scenario
            nndto = ledto.nndto_list[idx]
            if mode=='LFfromLE':
                v_known = ledto.v_ind_known[idx]
                X_pr, y_pr, _, _ = select_feature_label_scaled_scenario(pq_known,v_known,dataset,'LE',known_buses_ind)
                _ , best_model_path = get_best_model(nndto.results_df,nndto.results_models_paths,True)
                nndto.best_model = load_specific_models(results_path / le_model_dir,best_model_path)
                # for now only the first model is used for the prediction
                prediction_LE = nndto.best_model[0].predict(X_pr.values)
                dataset.pq_df = replace_columns_with_prediction(dataset,prediction_LE,y_pr.columns)
                pq_known = range(len(dataset.pq_df.columns))   # all loads are now known
                X_le, y_le, scaler_y, _ = select_feature_label_scaled_scenario(pq_known,v_known,dataset,'DLF',known_buses_ind)
            elif mode in ['LE','DLEF','DLF','BusLF']:
                v_known = ledto.v_ind_known[idx]
                X_le, y_le, scaler_y, scaler_X = select_feature_label_scaled_scenario(pq_known,v_known,dataset,mode,known_buses_ind, eval=True)
            elif (mode=='LF'):
                X_le, y_le, scaler_y = select_feature_label_scaled_LF(dataset)   #known_buses are not included here

            X_le_split = X_le.values
            y_le_split = y_le.values
            nndto.scalers = (None,scaler_y)

            Xtrain,Xtest,ytrain,ytest = train_test_split(X_le_split,y_le_split,test_size=0.15,random_state=8,shuffle=True)
            nndto.Xtest = pd.DataFrame(data=Xtest,columns=X_le.columns)
            nndto.ytest = pd.DataFrame(data=ytest,columns=y_le.columns)

            nndto.fraction_known = int(len(scenario))/(dataset_dict[name].pq_df.shape[1]) if mode!='LF' else 1
            nndto.df_avg, nndto.best_model_path = get_best_model(nndto.results_df,nndto.results_models_paths,True)
            nndto.best_model = load_specific_models(run_path,nndto.best_model_path)

            reg = LinearRegression(normalize=True)
            reg.fit(Xtrain,ytrain)
            nndto.model_linreg = reg
            if True:
                joblib.dump(reg, run_path / f"lin_reg_model_{idx}.pkl")

            ledto.nndto_list[idx] = nndto

            if X is not None:

                if dataset_eval is not None:
                    X_le, y_le, scaler_y_dummy, _dummy = select_feature_label_scaled_scenario(pq_known, v_known, dataset_eval, mode,
                                                                                   known_buses_ind, eval=True, training_scalerX=scaler_X, training_scalery=scaler_y, columns_in_order=[X_le.columns, y_le.columns])
                    X = X_le
                    y = y_le
                    X_scaled = X_le
                else:
                    y = X[1]
                    X = X[0]

                    columns = X_le.columns
                    X_scaled = pd.DataFrame(index=X.index)
                    for column in columns:
                        X_scaled[column] = scaler_X[column][0].transform(X[column].values.reshape(-1,1))       #voltages scales very weirdly; also powers don't look right


                #make predictions using all best models and average over results
                if not config.average_estimation_results:
                    index_best_model = nndto.results_df['metric'].idxmin()[1]
                    ypredict_nn_scaled = nndto.best_model[index_best_model-1].predict(X_scaled)    #which model best?
                else:
                    ypredict_nn_scaled_dict = {}
                    ypredict_nn_scaled_df = pd.DataFrame(index=X.index)
                    for model in nndto.best_model:
                        ypredict_nn_scaled_dict[model] = model.predict(X_scaled)
                    i = 1
                    for load in scaler_y.columns:

                        preds = pd.DataFrame(index=X.index)
                        n = 1
                        for pred in ypredict_nn_scaled_dict.items():
                            preds[str(n)] = pred[1][:,i-1:i]
                            n += 1

                        column = preds.values.mean(axis=1)
                        ypredict_nn_scaled_df[load] = column
                        i += 1
                    ypredict_nn_scaled = ypredict_nn_scaled_df.values

                ypredict_lr_scaled = nndto.model_linreg.predict(X_scaled)
                ypredict_nn = pd.DataFrame(index=X.index)
                ypredict_lr = pd.DataFrame(index=X.index)
                y_unscaled = pd.DataFrame(index=X.index)
                i = 1
                y_scaled = pd.DataFrame(index=X.index)
                for column in scaler_y.columns:
                    ypredict_nn[column] = scaler_y[column][0].inverse_transform(ypredict_nn_scaled[:,i-1:i])
                    ypredict_lr[column] = scaler_y[column][0].inverse_transform(ypredict_lr_scaled[:,i-1:i])
                    y_unscaled[column] = scaler_y[column][0].inverse_transform(y[column].values.reshape(-1,1))
                    y_scaled[column] = scaler_y[column][0].transform(y[column].values.reshape(-1,1))
                    i += 1

                if config is not None:
                    if config.print_loss_and_y_vs_pred:
                        if dataset_eval is not None:
                            print(f'Prediction MSE NN: {mean_squared_error(ypredict_nn_scaled, y)}')
                            print(f'Prediction MSE LR: {mean_squared_error(ypredict_lr_scaled, y)}')
                        else:
                            print(f'Prediction MSE NN: {mean_squared_error(ypredict_nn_scaled, y_scaled)}')
                            print(f'Prediction MSE LR: {mean_squared_error(ypredict_lr_scaled, y_scaled)}')

                        print(f'prediction: {ypredict_nn}')
                        print(f'target: {y_unscaled}')

                y_pred_nn = ypredict_nn[::pad_factor]
                y_pred_lr = ypredict_lr[::pad_factor]
                if dataset_eval is not None:
                    y = y_unscaled
                    y_plot= y[::pad_factor]
                else:
                    y = y
                    y_plot= y[::pad_factor]
                if config.plot_real_vs_estimate:
                    plot_estimate_vs_target_by_load(y_plot,y_pred_nn,y_pred_lr, style='-.', phase=dir_results.split('_')[-3], setup=setup)

                return {'NN estimate': ypredict_nn, 'LR estimate': ypredict_lr}, y

        # create a metrics dataframe
        ledto.metrics_summary_df = pd.DataFrame()
        for idx,nndto in enumerate(ledto.nndto_list):

            # get metric_df for trained NN model
            # columns are: 'ycolumn','metric','r2','metric_sc','r2_sc' and 'metric_var','r2_var' if cross-validated
            if len(nndto.best_model)==1:
                metric_df = get_metrics_df(nndto.scalers[1].columns,nndto,nndto.best_model,metric=mean_squared_error)
            else:
                metric_dfs = []
                for model in nndto.best_model:
                    metric_dfs.append(get_metrics_df(nndto.scalers[1].columns,nndto,model,metric=mean_squared_error))
                metric_df = metrics_df_avg(metric_dfs)

            # get metric_linreg_df for trained linear regression model
            metric_linreg_df = get_metrics_df(nndto.scalers[1].columns,nndto,nndto.model_linreg,metric=mean_squared_error)
            metric_linreg_df.columns = ['ycolumn','metric_LR','r2_LR','metric_sc_LR','r2_sc_LR']

            metric_df = metric_df.merge(metric_linreg_df, on='ycolumn')

            if metric_rmse:
                try:
                    metric_df['metric_var'] = 0.25/metric_df['metric']*metric_df['metric_var']  #1st order approx.
                except: pass
                metric_df['metric'] = np.sqrt(metric_df['metric'])
                metric_df['metric_sc'] = np.sqrt(metric_df['metric_sc'])
                metric_df['metric_LR'] = np.sqrt(metric_df['metric_LR'])
                metric_df['metric_sc_LR'] = np.sqrt(metric_df['metric_sc_LR'])
            nndto.metric_df = metric_df


            ledto.metrics_summary_df = ledto.metrics_summary_df.append({'scenario': ledto.pq_ind_known[idx]},ignore_index=True)
            last_idx = ledto.metrics_summary_df.last_valid_index()
            ledto.metrics_summary_df.loc[last_idx,'fraction'] =  nndto.fraction_known

            if (mode=='LE')  or (mode=='DLEF'):
                # select the rows relating to P or Q loads:
                metric_R2_df_P = nndto.metric_df.loc[nndto.metric_df.ycolumn.str.contains('.+_P'),:]
                metric_R2_df_Q = nndto.metric_df.loc[nndto.metric_df.ycolumn.str.contains('.+_Q'),:]

                # NN
                ledto.metrics_summary_df.loc[last_idx,'metric_avg_P'] = metric_R2_df_P.loc[:,'metric'].mean()
                ledto.metrics_summary_df.loc[last_idx,'metric_avg_Q'] =  metric_R2_df_Q.loc[:,'metric'].mean()
                try:
                    ledto.metrics_summary_df.loc[last_idx,'metric_var_avg_P'] = metric_R2_df_P.loc[:,'metric_var'].mean()
                    ledto.metrics_summary_df.loc[last_idx,'metric_var_avg_Q'] =  metric_R2_df_Q.loc[:,'metric_var'].mean()
                except: pass
                ledto.metrics_summary_df.loc[last_idx,'metric_sc_avg_P'] =  metric_R2_df_P.loc[:,'metric_sc'].mean()
                ledto.metrics_summary_df.loc[last_idx,'metric_sc_avg_Q'] =  metric_R2_df_Q.loc[:,'metric_sc'].mean()
                ledto.metrics_summary_df.loc[last_idx,'r2_avg_P'] =  metric_R2_df_P.loc[:,'r2'].mean()
                ledto.metrics_summary_df.loc[last_idx,'r2_avg_Q'] =  metric_R2_df_Q.loc[:,'r2'].mean()
                try:
                    ledto.metrics_summary_df.loc[last_idx,'r2_var_avg_P'] =  metric_R2_df_P.loc[:,'r2_var'].mean()
                    ledto.metrics_summary_df.loc[last_idx,'r2_var_avg_Q'] =  metric_R2_df_Q.loc[:,'r2_var'].mean()
                except: pass
                ledto.metrics_summary_df.loc[last_idx,'r2_sc_avg_P'] =  metric_R2_df_P.loc[:,'r2_sc'].mean()
                ledto.metrics_summary_df.loc[last_idx,'r2_sc_avg_Q'] =  metric_R2_df_Q.loc[:,'r2_sc'].mean()
                # linear regression
                ledto.metrics_summary_df.loc[last_idx,'metric_avg_P_LR'] = metric_R2_df_P.loc[:,'metric_LR'].mean()
                ledto.metrics_summary_df.loc[last_idx,'metric_avg_Q_LR'] =  metric_R2_df_Q.loc[:,'metric_LR'].mean()
                ledto.metrics_summary_df.loc[last_idx,'metric_sc_avg_P_LR'] =  metric_R2_df_P.loc[:,'metric_sc_LR'].mean()
                ledto.metrics_summary_df.loc[last_idx,'metric_sc_avg_Q_LR'] =  metric_R2_df_Q.loc[:,'metric_sc_LR'].mean()
                ledto.metrics_summary_df.loc[last_idx,'r2_avg_P_LR'] =  metric_R2_df_P.loc[:,'r2_LR'].mean()
                ledto.metrics_summary_df.loc[last_idx,'r2_avg_Q_LR'] =  metric_R2_df_Q.loc[:,'r2_LR'].mean()
                ledto.metrics_summary_df.loc[last_idx,'r2_sc_avg_P_LR'] =  metric_R2_df_P.loc[:,'r2_sc_LR'].mean()
                ledto.metrics_summary_df.loc[last_idx,'r2_sc_avg_Q_LR'] =  metric_R2_df_Q.loc[:,'r2_sc_LR'].mean()

            if (mode=='DLF') or (mode=='LF') or (mode=='DLEF') or (mode=='BusLF') or (mode=='LFfromLE'):

                # select the row relating to buses:
                metric_R2_df_bus = nndto.metric_df.loc[nndto.metric_df.ycolumn.str.contains('.+_V'),:]
                #optional: exclude buses from evaluation
                if len(exlude_buses_in_evaluation) > 0:
                    selector = metric_R2_df_bus.ycolumn.isin(exlude_buses_in_evaluation)
                    metric_R2_df_bus = metric_R2_df_bus.loc[~selector,:]

                # NN
                ledto.metrics_summary_df.loc[last_idx,'metric_avg_bus'] =    metric_R2_df_bus.loc[:,'metric'].mean()
                ledto.metrics_summary_df.loc[last_idx,'metric_sc_avg_bus'] = metric_R2_df_bus.loc[:,'metric_sc'].mean()
                ledto.metrics_summary_df.loc[last_idx,'r2_avg_bus'] =        metric_R2_df_bus.loc[:,'r2'].mean()
                ledto.metrics_summary_df.loc[last_idx,'r2_sc_avg_bus'] =     metric_R2_df_bus.loc[:,'r2_sc'].mean()
                try:
                    ledto.metrics_summary_df.loc[last_idx,'metric_var_avg_bus'] =    metric_R2_df_bus.loc[:,'metric_var'].mean()
                    ledto.metrics_summary_df.loc[last_idx,'r2_var_avg_bus'] =        metric_R2_df_bus.loc[:,'r2_var'].mean()
                except: pass
                # linear regression
                ledto.metrics_summary_df.loc[last_idx,'metric_avg_bus_LR'] =    metric_R2_df_bus.loc[:,'metric_LR'].mean()
                ledto.metrics_summary_df.loc[last_idx,'metric_sc_avg_bus_LR'] = metric_R2_df_bus.loc[:,'metric_sc_LR'].mean()
                ledto.metrics_summary_df.loc[last_idx,'r2_avg_bus_LR'] =        metric_R2_df_bus.loc[:,'r2_LR'].mean()
                ledto.metrics_summary_df.loc[last_idx,'r2_sc_avg_bus_LR'] =     metric_R2_df_bus.loc[:,'r2_sc_LR'].mean()

        print(ledto.metrics_summary_df)
        save_ledto_metrics_summary(ledto,path=run_path,name=f"grid{name}")

    # Plot the load estimation results
    for name,ledto in results_dict.items():
        for idx,nndto in enumerate(ledto.nndto_list):
            if (mode == 'DLF') or (mode=='LF') or (mode=='BusLF') or (mode == 'DLEF') or (mode=='LFfromLE'):
                #plot_predict_voltage(nndto,buses=pred_plot_buses,save=save_fig_scen,path=run_path,minx=2000,maxx=2100)
                plot_actual_pred_voltage(nndto,buses=None,save=save_fig_scen,path=run_path)
                plot_residuals_voltage(nndto,buses=None,save=save_fig_scen,path=run_path)
                #pass
        if (mode=='LE') or (mode=='DLEF'):
            plot_metric_R2_scenarios(results_dict[name],dataset_dict[name],metric_name='RMSE [kW]',save=save_fig_scen,fig_path=run_path,name=f"grid{name}",metric_factor=1000)
            plot_metric_R2_fraction(results_dict[name],metric_name='RMSE [kW]',save=save_fig_frac,fig_path=run_path,name=f"for grid{name}",metric_factor=1000)
            plot_metric_R2_fraction_LR(results_dict[name],metric_name='RMSE [kW]',save=save_fig_frac,fig_path=run_path,name=f"for grid{name}",metric_factor=1000)
        if (mode == 'DLF') or (mode=='LF') or (mode=='BusLF') or (mode == 'DLEF') or (mode=='LFfromLE'):
            plot_metric_R2_scenarios_buses(results_dict[name],dataset_dict[name],exlude_buses_in_evaluation,metric_name='RMSE [pu]',save=save_fig_scen,fig_path=run_path,name=f"grid{name}",metric_factor=1)
            plot_metric_R2_fraction_bus(results_dict[name],metric_name='RMSE [pu]',save=save_fig_scen,fig_path=run_path,name=f"for grid{name}",metric_factor=1)
            plot_metric_R2_fraction_general(results_dict[name].metrics_summary_df,
                            ycolsR2 = ['r2_avg_bus','r2_avg_bus_LR'],
                            ycolMet = ['metric_avg_bus', 'metric_avg_bus_LR'],
                            ycolors = [ait_colors['türkis'],ait_colors['grau']],
                            ylabels = ['Bus','Bus - LR'],
                            title = 'LR -',
                            metric_name='RMSE [pu]',
                            save=save_fig_frac,
                            fig_path=run_path,
                            name=f"for grid{name}",
                            metric_factor=1)