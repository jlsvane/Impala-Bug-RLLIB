#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 22:05:30 2023

@author: lupus
"""

import numpy as np
import gymnasium
from gymnasium.spaces import Discrete, Box
from ray.tune.registry import register_env
# import ray.rllib.algorithms.impala as impala
import ray.rllib.algorithms.ppo as ppo
from ray.tune.logger import pretty_print
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models import ModelCatalog

tf1, tf, tfv = try_import_tf()


def conv_layer(depth, name):
    return tf.keras.layers.Conv2D(
        filters=depth, kernel_size=3, strides=1, padding="same", name=name
    )


def residual_block(x, depth, prefix):
    inputs = x
    assert inputs.get_shape()[-1].value == depth
    x = tf.keras.layers.ReLU()(x)
    x = conv_layer(depth, name=prefix + "_conv0")(x)
    x = tf.keras.layers.ReLU()(x)
    x = conv_layer(depth, name=prefix + "_conv1")(x)
    return x + inputs


def conv_sequence(x, depth, prefix):
    x = conv_layer(depth, prefix + "_conv")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=3, strides=2, padding="same")(x) # 3D for multiframe
    x = residual_block(x, depth, prefix=prefix + "_block0")
    x = residual_block(x, depth, prefix=prefix + "_block1")
    return x


class CustomModel(TFModelV2):
    """Deep residual network that produces logits for policy and value for value-function;
    Based on architecture used in IMPALA paper:https://arxiv.org/abs/1802.01561"""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        depths = [16, 32, 32]

        inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        scaled_inputs = tf.cast(inputs, tf.float32) / 255.0

        x = scaled_inputs
        for i, depth in enumerate(depths):
            x = conv_sequence(x, depth, prefix=f"seq{i}")

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(units=256, activation="relu", name="hidden")(x)
        logits = tf.keras.layers.Dense(units=num_outputs, name="pi")(x)
        value = tf.keras.layers.Dense(units=1, name="vf")(x)
        self.base_model = tf.keras.Model(inputs, [logits, value])

    def forward(self, input_dict, state, seq_lens):
        # explicit cast to float32 needed in eager
        obs = tf.cast(input_dict["obs"], tf.float32)
        logits, self._value = self.base_model(obs)
        return logits, state

    def value_function(self):
        return tf.reshape(self._value, [-1])
    
    def import_from_h5(self, h5_file):
        self.base_model.load_weights(h5_file)

config = {
        "observation_space": Box(
            low=0,
            high=255,
            shape=(72, 128, 3),
            dtype=np.uint8),
        "action_space": Discrete(5),
        "p_terminated": 1e-4, # to prevent early termination - decrease if needed
        "max_episode_len":495,
        "sleeping":1.0 # to mimic slow env - increase if needed
        }

class RewardWrapper(gymnasium.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def reward(self, rew):
        # downscale and clip reward
        rew_downscaled = np.clip(rew/100,-1.0,1.0)
        return rew_downscaled


def env_creator(env_config):
    kwargs = {}
    env = gymnasium.make('my_random:my_random/MyRandomEnv-v0',config=env_config,**kwargs)
    env = RewardWrapper(env)
    env = gymnasium.wrappers.FrameStack(env, 4)
    return env

register_env("myrandomenv", env_creator)

# Register custom-model in ModelCatalog
ModelCatalog.register_custom_model("CustomCNN", CustomModel)

algo = (
    ppo.PPOConfig()
    .training(
        lr=5e-4,
        lr_schedule=[[0, 5e-4],[200e6, 0.0]],
        vf_loss_coeff=0.5,
        train_batch_size=2400,
        model={"custom_model":"CustomCNN"},
        entropy_coeff=5e-3,
        entropy_coeff_schedule=[[0, 5e-3],[100e6, 1e-3],[200e6, 5e-5]],
        grad_clip=40.0) 
    .environment(env="myrandomenv",env_config=config)
    .framework(framework="tf2",eager_tracing=True)
    .rollouts(
              num_rollout_workers=6, # 6 for single machine
              num_envs_per_worker=4,
              rollout_fragment_length=100,
              remote_worker_envs=True,
              remote_env_batch_wait_ms=10,
              preprocessor_pref=None,
              # sampler_perf_stats_ema_coef=2/(200+1) # 200 ema
               )
    .resources(num_gpus=1)
    # .resources(num_gpus=1,num_cpus_per_worker=5)
    # .fault_tolerance(recreate_failed_workers=True, restart_failed_sub_environments=True)
    
    .build()
)

for i in range(100):
    result = algo.train()
    print(pretty_print(result))

    # if i % 5 == 0:
    #     checkpoint_dir = algo.save("./ray_231_test")
    #     print(f"Checkpoint saved in directory {checkpoint_dir}")