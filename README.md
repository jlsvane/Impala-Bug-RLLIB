# Impala-Bug-RLLIB

Reproduction code for rllib Impala bug

1. Clone
2. pip install my-random
3. run impala_test_random_env_costum_model.py

The problem with the rllib Impala algorithm appears to be caused by a slow environment. 
Depending on system you may wan't to play with the 'sleeping' parameter as well as num_rollout_workers etc.
