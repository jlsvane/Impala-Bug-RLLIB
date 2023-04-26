# Impala-Bug-RLLIB

Reproduction code for rllib Impala bug

1. Clone
2. pip install my-random or cd my-random and then pip install -e .
3. run impala_test_random_env_costum_model.py
4. run impala_test_random_env_custom_model_231.py for testing in Ray 2.3.1
5. run any of the appo or ppo reference files as you see fit

The problem with the rllib Impala algorithm appears to be caused by a slow environment. 
Depending on system you may wan't to play with the 'sleeping' parameter as well as num_rollout_workers etc.
