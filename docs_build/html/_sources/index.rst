=====================
PdMLabs documentation
=====================

**PdMLabs is a free and open-source Python library** created to help data scientists identify **early signs before events** of interest and model **time-to-event prediction**.

.. grid:: 1 2 2 2
    :gutter: 4
    :padding: 2 2 0 0
    :class-container: sd-text-center

    .. grid-item-card:: 📜 Our Manifesto
        :link: introduction
        :link-type: doc
        :class-header: bg-light

        Understand what PdMLabs is all about

    .. grid-item-card:: 🚀 Quick Start
        :link: getting-started/quickstart
        :link-type: doc
        :class-header: bg-light

        Get up and running in less than 5 minutes

    .. grid-item-card:: 💡 Examples
        :link: examples/index
        :link-type: doc
        :class-header: bg-light

        Explore real-world examples and use cases

    .. grid-item-card:: 📚 Concepts
        :link: concepts/index
        :link-type: doc
        :class-header: bg-light

        Learn about Experiments, Methodologies, and more

    .. grid-item-card:: 📖 API Reference
        :link: api-reference
        :link-type: doc
        :class-header: bg-light

        Complete API documentation

Installation
------------

.. code-block:: bash

    pip install pdmlabs

**Requirements**: Python >= 3.11


TL;DR
-----

Load your data:

.. code-block:: python

   from pdmlabs.utils.dataset import Dataset
   df=pd.read_csv("data/ims.csv")

   dataset_handler = Dataset(df,datetime_column="timestamp",train_sources=0.6,val_sources=0.2,test_sources=0.2)
   Train_Val_data, Train_Test_data = dataset_handler.get_unsupervised_dataset() # for early signs detection using anomaly detection

   # Depending on the goal of your analysis, you can also get the data for different modeling approaches:
   # Train_Val_data, _ = dataset_handler.get_rul_dataset() # for early signs detection using semi-supervised anomaly detection
   # Train_Val_data, _ = dataset_handler.get_rul_dataset() # for early signs detection using classification
   # Train_Val_data, _ = dataset_handler.get_rul_dataset() # for time-to-event prediction using regression
   # Train_Val_data, _ = dataset_handler.get_unsupervised_dataset() # for time-to-event prediction using Survival Analysis


Perform your experiment (For guidance on how to choose the right experiment for your case, check the :doc:`concepts/index`):


.. code-block:: python

   from pdmlabs.experiment.batch.auto_profile_semi_supervised_experiment import AutoProfileSemiSupervisedPdMExperiment
   from pdmlabs.RunExperiment import run_experiment
   # available Anomaly detection experiments: AutoProfileSemiSupervisedPdMExperiment,IncrementalSemiSupervisedPdMExperiment,UnsupervisedPdMExperiment,SemiSupervisedPdMExperiment
   # Classification experiments: SupervisedPdMExperiment
   # Time-to-event regression experiments: SupervisedRULPdMExperiment
   # Time-to-event with Survival Analysis experiments: Supervised_SA_PdMExperiment

   experiments = [AutoProfileSemiSupervisedPdMExperiment]
   experiment_names = ['My first experiment']
   fit_size=1000 # initial data to fit, this is specific to AutoProfileSemiSupervisedPdMExperiment

   from pdmlabs.method.isolation_forest import IsolationForest
   from pdmlabs.method.lof_semi import LocalOutlierFactor

   methods = [IsolationForest,LocalOutlierFactor]
   param_space_dict_per_method = [{'n_estimators': [200,100],'max_samples': [200,100],'random_state': [42],
                                   'max_features': [0.8,0.5],'bootstrap': [True,False]},
                                   {'n_neighbors':[2,3,5,10,20]}]

   method_names = ["IF","LOF"]


   run_experiment(Train_Val_data, methods, param_space_dict_per_method, method_names,
                       experiments, experiment_names, mlflow_port=5000,
                       MAX_RUNS=4, MAX_JOBS=1, INITIAL_RANDOM=1, optimization_param="AD1_AUC", debug=True,
                       maximize=maximize,profile_size=profile_for_test)


.. code-block:: text

   Best score: 0.5741854636591478: 100%|██████████| 3/3 [00:03<00:00,  1.11s/it]
   My first experiment IF
   {'best_params': {'init_profile_size': 1000, 'method_bootstrap': True, 'method_max_features': 0.8, 'method_max_samples': 200, 'method_n_estimators': 100, 'method_random_state': 42, 'profile_size': 2}, 'best_objective': 0.5741854636591478}

   Best score: 0.6392039: 100%|██████████| 3/3 [00:03<00:00,  1.11s/it]
   My first experiment LOF
   {'best_params': {'n_neighbors': 8}, 'best_objective': 0.6392039}

   Process finished with exit code 0


Key Features
-------------

.. grid:: 1 1 2 3
    :gutter: 4
    :padding: 2 2 0 0

    .. grid-item-card:: ⚡
        :class-header: bg-primary text-white

        Get instant alerts when your data quality drops below your defined thresholds

    .. grid-item-card:: Flexible
        :class-header: bg-primary text-white

        Choose from Classification, multiple Anomaly Detection flavors, Regression, and Survival Analysis modeling for finding early signs before events of interest or predicting time to event.

    .. grid-item-card:: Automated
        :class-header: bg-primary text-white

        Experiments leverage Bayesian Optimization to find the best model and hyperparameters for your data

    .. grid-item-card:: Extensible
        :class-header: bg-primary text-white

        Add custom models by implementing a simple interface.

    .. grid-item-card:: 📊 Rich Output
        :class-header: bg-primary text-white

        Keep track of experiments and check results using Mlflow dashboard.

.. admonition:: Perfect for
   :class: tip

   - **Predictive Maintenance** monitoring sensor data for early signs of equipment failure
   - **Data Scientists** building robust models for time-to-event prediction
   - **Analytics Teams** seeking automated generation of insights from experimental analysis results

   The framework is particularly useful when you need to:

   - **Identify the most suitable methodology for a specific PdM case**, especially when the appropriate modeling approach or technique flavor is not obvious.
   - **Discover algorithm configurations that meet user-defined performance thresholds**, enabling the use of interpretable methods without sacrificing predictive capability.
   - **Analyze trade-offs between predictive performance and computational efficiency**, which is crucial when monitoring large fleets of assets or operating under resource or environmental constraints.
   - **Perform comparative evaluations of alternative solutions**, helping determine whether investing in a new method provides meaningful improvements over existing approaches.


Next Steps
-----------

.. toctree::
   :hidden:
   :maxdepth: 1

   🏠 Home <self>

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contents

   introduction
   concepts/index
   getting-started/index
   examples/index
   user-guide/index
   api-reference

Ready to dive in? Here are some suggested paths:

**New to PdMLabs?** → Start with :doc:`introduction`

**Ready for action?** → Jump straight to the :doc:`getting-started/quickstart`

**Eager to deepen understanding?** → Read :doc:`concepts/index`

**Looking for examples?** → Check out :doc:`examples/index`

**Need detailed configuration?** → Browse :doc:`user-guide/index`


Acknowledgments
---------------

PdMLabs is developed by Anastasios Papadopoulos and Apostolos Giannoulidis and supported by the Data Engineering (DELAB) Team of `Datalab AUTh <https://datalab.csd.auth.gr/>`_,
under the supervision of `Prof. Anastasios Gounaris <https://datalab-old.csd.auth.gr/~gounaris/>`_.

PdMLabs incorporates RUL and Survival Analysis evaluation components from the `TITEUF project <https://github.com/agiannoul/TITEUF>`_.
Licensed under the Apache License, Version 2.0.
