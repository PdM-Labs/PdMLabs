import os
import sys
import pandas as pd
import mlflow

# Add root project path to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pdmlabs.utils.dataset import Dataset
from pdmlabs.RunExperiment import run_experiment
from pdmlabs.experiment.batch.supervised_experiment import SupervisedPdMExperiment
from pdmlabs.method.xgboost import XGBoost

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
    
    print("Preparing Classification Dataset format...")
    train_data, test_data = dataset_obj.get_Classification_dataset()
    
    print("Running Experiment (Supervised)...")
    methods = [XGBoost]
    param_spaces = [{'n_estimators': [10]}] # Dummy fast param space
    
    best_params = run_experiment(
        dataset=train_data,
        methods=methods,
        param_space_dict_per_method=param_spaces,
        method_names=['XGBoost'],
        experiments=[SupervisedPdMExperiment],
        experiment_names=['Test_Supervised'],
        MAX_RUNS=1,
        MAX_JOBS=1,
        INITIAL_RANDOM=1,
        optimization_param="AD1_AUC"
    )
    
    print("Experiment finished. Best params:", best_params)
    
    print("Verifying MLflow tracking and Model loading...")
    experiment_name = 'Test_Supervised XGBoost'
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
        
    print("Supervised Test Successful.")

if __name__ == '__main__':
    main()
