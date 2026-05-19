import os
import sys
import pandas as pd
import mlflow

# Add root project path to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pdmlabs.utils.dataset import Dataset
from pdmlabs.RunExperiment import run_experiment
from pdmlabs.experiment.batch.incremental_semi_supervised_experiment import IncrementalSemiSupervisedPdMExperiment
from pdmlabs.method.ocsvm import OneClassSVM

def main():
    print("Loading ims_all.csv...")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'ims_all.csv')
    df = pd.read_csv(data_path)
    
    print("Initializing Dataset...")
    dataset_obj = Dataset(
        data=df,
        datetime_column='Artificial_timestamp',
        event_indicator='event',
        source_column='source',
        train_sources=['1'],
        val_sources=['2'],
        test_sources=['3']
    )
    
    print("Preparing Semi Dataset format...")
    train_data, test_data = dataset_obj.get_semi_dataset()
    
    print("Running Experiment (Incremental Semi-Supervised)...")
    methods = [OneClassSVM]
    param_spaces = [{'nu': [0.1]}]
    additional_parameters={'initial_incremental_window_length': [100], 'incremental_window_length': [100], 'incremental_slide': [100]}
    
    best_params = run_experiment(
        dataset=train_data,
        methods=methods,
        param_space_dict_per_method=param_spaces,
        method_names=['OneClassSVM'],
        experiments=[IncrementalSemiSupervisedPdMExperiment],
        experiment_names=['Test_Incremental'],
        additional_parameters=additional_parameters,
        MAX_RUNS=1,
        MAX_JOBS=1,
        INITIAL_RANDOM=1,
        optimization_param="AD1_AUC"
    )
    
    print("Experiment finished. Best params:", best_params)
    
    print("Verifying MLflow tracking and Model loading...")
    experiment_name = 'Test_Incremental OneClassSVM'
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        raise RuntimeError(f"MLflow experiment '{experiment_name}' not found!")
        
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
    run_id = runs.iloc[0].run_id
    
    print(f"Loading best pipeline from run ID: {run_id}")
    loaded_pipeline = mlflow.pyfunc.load_model(f"runs:/{run_id}/best_pdm_pipeline")
    
    print("Testing loaded pipeline on Test Source '3'...")
    for target_data, target_source in zip(test_data['target_data'], test_data['target_sources']):
        inference_data = target_data.copy()
        if 'Artificial_timestamp' in inference_data.columns:
            inference_data = inference_data.drop(columns=['Artificial_timestamp'])
        preds = loaded_pipeline.predict({'target_data': inference_data, 'source': target_source, 'event_data': test_data['event_data']})
        print(f"Predictions successfully generated for source {target_source}! Size: {len(preds['scores'])}")
        
    print("Incremental Semi-Supervised Test Successful.")

if __name__ == '__main__':
    main()
