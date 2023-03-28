# NPU (华为 昇腾)

## 使用方法

**步骤 0.** 安装MMCV：请参考 [MMCV 的安装文档](https://mmcv.readthedocs.io/en/latest/get_started/build.html#build-mmcv-full-on-ascend-npu-machine) 来安装 NPU 版本的 MMCV。

**步骤 1.** 安装 MMDetection：请参考[MMDetection的安装文档](https://mmdetection.readthedocs.io/zh_CN/latest/get_started.html#mmdetection)来进行安装。

**步骤 2.** 安装 MMSegmentation：请参考[MMSegmentation的安装文档](https://mmsegmentation.readthedocs.io/zh_CN/latest/get_started.html#id2)来进行安装。

以下展示单机八卡场景的运行指令:

```shell
bash tools/dist_train.sh configs/centerpoint/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus.py 8
```

以下展示单机单卡下的运行指令:

```shell
python tools/train.py configs/centerpoint/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus.py
```

## 模型验证结果

|        Model        | box AP | Config                                                                                                                                         | Download             |
| :-----------------: | :----: | :--------------------------------------------------------------------------------------------------------------------------------------------- | :------------------- |
| [centerpoint\*](<>) |  48.6  | [config](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/centerpoint/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus.py) | [log](<>)   **todo** |

**注意:**

- (\*) 当在NPU上运行centerpoint模型时，需要进行一定的修改：

  **修改原因：** NPU上的pad算子当前不支持填充int32值，计划在2023年Q2完成修复。

  **修改方式：** 使用cat算子替换pad算子。

  **修改函数：** mmdet3d/models/detectors/mvx_two_stage.py  voxelize函数。

  **修改前：**

  ```python
  def voxelize(self, points):
      """Apply dynamic voxelization to points.

      Args:
          points (list[torch.Tensor]): Points of each sample.

      Returns:
          tuple[torch.Tensor]: Concatenated points, number of points
              per voxel, and coordinates.
      """
      voxels, coors, num_points = [], [], []
      for res in points:
          res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
          voxels.append(res_voxels)
          coors.append(res_coors)
          num_points.append(res_num_points)
      voxels = torch.cat(voxels, dim=0)
      num_points = torch.cat(num_points, dim=0)
      coors_batch = []
      for i, coor in enumerate(coors):
          coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
          coors_batch.append(coor_pad)
      coors_batch = torch.cat(coors_batch, dim=0)
      return voxels, num_points, coors_batch
  ```

  **修改后：**

  ```python
  def voxelize(self, points):
   """Apply dynamic voxelization to points.

   Args:
       points (list[torch.Tensor]): Points of each sample.

   Returns:
       tuple[torch.Tensor]: Concatenated points, number of points
           per voxel, and coordinates.
   """
   voxels, coors, num_points = [], [], []
   for res in points:
       res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
       voxels.append(res_voxels)
       coors.append(res_coors)
       num_points.append(res_num_points)
   voxels = torch.cat(voxels, dim=0)
   num_points = torch.cat(num_points, dim=0)
   coors_batch = []
   for i, coor in enumerate(coors):
       pad_value = torch.full([coor.shape[0], 1], fill_value=i).to(
           coor.device).to(coor.dtype)
       coor_pad = torch.cat([pad_value, coor], dim=1)
       coors_batch.append(coor_pad)
   coors_batch = torch.cat(coors_batch, dim=0)
   return voxels, num_points, coors_batch
  ```

**以上模型结果由华为昇腾团队提供**
