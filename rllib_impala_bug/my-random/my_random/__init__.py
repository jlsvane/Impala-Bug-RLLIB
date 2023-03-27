#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:23:13 2023

@author: lupus
"""

from gymnasium.envs.registration import register

register(
    id="my_random/MyRandomEnv-v0",
    entry_point="my_random.envs:MyRandomEnv",
)
