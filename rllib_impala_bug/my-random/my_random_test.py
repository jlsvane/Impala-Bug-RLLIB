#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 13:38:39 2023

@author: novelty
"""

import gymnasium
import my_random
env = gymnasium.make('my_random/MyRandomEnv-v0')

state, _ = env.reset()

for i in range(10):
    
    state, reward, terminated, truncated, _ = env.step(env.action_space.sample())