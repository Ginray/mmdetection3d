# NPU (HUAWEI Ascend)

## Usage

**Step 0.** Install MMCV: Please refer to the [building documentation of MMCV](https://mmcv.readthedocs.io/en/latest/get_started/build.html#build-mmcv-full-on-ascend-npu-machine) to install MMCV on NPU devices.

**Step 1.** Install MMDetection：Please refer to the [building documentation of MMDetection](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation) to install.

**Step 2.** Install MMSegmentation：Please refer to the [building documentation of MMSegmentation](https://mmsegmentation.readthedocs.io/en/latest/get_started.html#installation) to install.

Here we use 8 NPUs on your computer to train the model with the following command:

```shell
bash tools/dist_train.sh configs/centerpoint/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus.py 8
```

Also, you can use only one NPU to train the model with the following command:

```shell
python tools/train.py configs/centerpoint/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus.py
```

## Models Results

|        Model        | box AP | Config                                                                                                                                         | Download             |
| :-----------------: | :----: | :--------------------------------------------------------------------------------------------------------------------------------------------- | :------------------- |
| [centerpoint\*](<>) |  48.6  | [config](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/centerpoint/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus.py) | [log](<>)   **todo** |

**All above models are provided by Huawei Ascend group.**
