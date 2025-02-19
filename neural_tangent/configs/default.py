# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.learning_rate = 0.01
    config.momentum = 0.9
    config.batch_size = 128
    config.num_epochs = 10
    config.num_rounds = 5

    return config

def metrics():
    return []