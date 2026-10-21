import os
import sys
import pandas as pd
import mlflow

# Add root project path to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pdmlabs.utils.dataset import Dataset
from pdmlabs.RunExperiment import run_experiment
from pdmlabs.experiment.batch.semi_supervised_experiment import SemiSupervisedPdMExperiment
from pdmlabs.method.ocsvm import OneClassSVM
from pdmlabs.preprocessing.raw_level.statistical import StatisticalRawPreProcessor
from pdmlabs.preprocessing.raw_level.snapshots import snapshot_frame


def main():
    print("Loading data/snapshots...")
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'snapshots')
    # Sources s1 and s2 are folders (one CSV file per snapshot); s3 is given as a list of
    # DataFrames, so both kinds of snapshot sources are exercised.
    s3_dir = os.path.join(data_dir, 's3')
    s3_frames = [pd.read_csv(os.path.join(s3_dir, name)) for name in sorted(os.listdir(s3_dir))]
    # The snapshot options apply to every source; 'datetime_column' works for folders and
    # DataFrames alike: each snapshot's time is the first value of its 'time' column.
    snapshot_options = {'datetime_column': 'time'}

    print("Initializing Dataset...")
    dataset_obj = Dataset(
        data={
            's1': os.path.join(data_dir, 's1'),
            's2': os.path.join(data_dir, 's2'),
            's3': s3_frames,
        },
        datetime_column='timestamp',
        source_column='source',
        train_sources=['s1'],
        val_sources=['s2'],
        test_sources=['s3'],
        snapshot_options=snapshot_options
    )

    print("Building the same Dataset from the root folder (every subfolder is a source)...")
    folder_dataset = Dataset(
        data=data_dir,
        datetime_column='timestamp',
        source_column='source',
        train_sources=['s1'],
        val_sources=['s2'],
        test_sources=['s3'],
        snapshot_options=snapshot_options
    )
    print("Sources found:", folder_dataset.sources)

    print("Preparing Semi Dataset format...")
    train_data, test_data = dataset_obj.get_semi_dataset()

    print("Running Experiment (Snapshots, Semi-Supervised)...")
    methods = [OneClassSVM]
    param_spaces = [{'nu': [0.1]}] # Dummy fast param space

    best_params = run_experiment(
        dataset=train_data,
        methods=methods,
        param_space_dict_per_method=param_spaces,
        method_names=['OneClassSVM'],
        experiments=[SemiSupervisedPdMExperiment],
        experiment_names=['Test_Snapshots'],
        raw_preprocessor=StatisticalRawPreProcessor,
        additional_parameters={'raw_preprocessor_n_bins': [2, 4]},
        MAX_RUNS=2,
        MAX_JOBS=1,
        INITIAL_RANDOM=1,
        optimization_param="AD1_AUC"
    )

    print("Experiment finished. Best params:", best_params)

    print("Verifying MLflow tracking and Model loading...")
    experiment_name = 'Test_Snapshots OneClassSVM'
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise RuntimeError(f"MLflow experiment '{experiment_name}' not found!")

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
    run_id = runs.iloc[0].run_id

    print(f"Loading best pipeline from run ID: {run_id}")
    loaded_pipeline = mlflow.pyfunc.load_model(f"runs:/{run_id}/best_pdm_pipeline")

    print("Testing loaded pipeline on Test Source 's3'...")
    # snapshot_frame turns the snapshots of one source into the input predict expects; the
    # pipeline's raw-level step computes their features.
    inference_data = snapshot_frame(s3_frames, **snapshot_options)
    preds = loaded_pipeline.predict({'target_data': inference_data, 'source': 's3', 'event_data': test_data['event_data']})
    print(f"Predictions successfully generated for source s3! Size: {len(preds['scores'])}")

    print("Testing loaded pipeline on 's4_xy', read with a reader...")
    # s4_xy was recorded with two sensors per channel (ch0_x, ch0_y, ch1_x, ch1_y), like IMS
    # test 1, so a reader keeps one sensor per channel, named as in the training data.
    def read_xy(path, **kwargs):
        frame = pd.read_csv(path, **kwargs)
        return frame[['time', 'ch0_x', 'ch1_x']].set_axis(['time', 'ch0', 'ch1'], axis=1)

    inference_data = snapshot_frame(os.path.join(data_dir, 's4_xy'), datetime_column='time', reader=read_xy)
    preds = loaded_pipeline.predict({'target_data': inference_data, 'source': 's4_xy', 'event_data': test_data['event_data']})
    print(f"Predictions successfully generated for source s4_xy! Size: {len(preds['scores'])}")

    print("Snapshots Test Successful.")

if __name__ == '__main__':
    main()
