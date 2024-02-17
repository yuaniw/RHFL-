# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import sys
import os
from mindspore.train.serialization import export

sys.path.append('..')
from Dataset.utils import init_logs, init_nets
import mindspore
import numpy as np
import yaml

f = open('../config.yaml', 'r', encoding='utf-8')
fdata = f.read()
f.close()
cfg = yaml.safe_load(fdata)

n_participants = cfg['n_participants'] 
pariticpant_params = cfg['rhfl']['pariticpant_params']

noise_type = cfg['noise_type'] 
noise_rate = cfg['noise_rate'] 
"""Heterogeneous Model Setting"""
private_nets_name_list = ['ResNet10', 'ResNet12', 'ShuffleNet', 'EfficientNet']



if __name__ == '__main__':
    logger = init_logs()

    net_list = init_nets(n_parties=n_participants, nets_name_list=private_nets_name_list)
    logger.info("Load Participants' Models")

    for i in range(n_participants):
        network = net_list[i]
        netname = private_nets_name_list[i]
        param_dict = mindspore.load_checkpoint('./test/Model_Storage/' +
                                               pariticpant_params['loss_funnction'] + '/' +
                                               str(noise_type) + str(noise_rate) + '/' +
                                               netname + '_' + str(i) + '.ckpt')
        mindspore.load_param_into_net(network, param_dict)
        input = np.random.uniform(0.0, 1.0, size=[32, 3, 224, 224]).astype(np.float32)
        export(network, mindspore.Tensor(input), file_name='./storage/Model_Storage/' +
                                               pariticpant_params['loss_funnction'] + '/' +
                                               str(noise_type) + str(noise_rate) + '/' +
                                               netname + '_' + str(i) + '.mindir', file_format='MINDIR')
        export(network, mindspore.Tensor(input), file_name='./storage/Model_Storage/' +
                                               pariticpant_params['loss_funnction'] + '/' +
                                               str(noise_type) + str(noise_rate) + '/' +
                                               netname + '_' + str(i) + '.air', file_format='AIR')
        logger.info(netname + '_' + str(i) + "model exported successfully")
        
    logger.info("Models exported successfully")
