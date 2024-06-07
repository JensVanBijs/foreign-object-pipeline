conda activate /data/s2902702/cito/astra-env
python /data/s2902702/cito/foreign-object-workflow/experiment_code/phantom_generation.py
python /data/s2902702/cito/foreign-object-workflow/experiment_code/object_projections.py
python /data/s2902702/cito/foreign-object-workflow/experiment_code/object_reconstruction.py
python /data/s2902702/cito/foreign-object-workflow/experiment_code/target_projections.py
python /data/s2902702/cito/foreign-object-workflow/experiment_code/model_training.py
python /data/s2902702/cito/foreign-object-workflow/experiment_code/model_testing.py
conda deactivate