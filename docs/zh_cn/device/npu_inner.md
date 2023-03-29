# NPU (华为 昇腾)

## 使用方法

### 测试步骤

#### 1、准备环境

**步骤 0.** 安装MMCV：请参考 [MMCV 的安装文档](https://mmcv.readthedocs.io/en/latest/get_started/build.html#build-mmcv-full-on-ascend-npu-machine) 来安装 NPU 版本的 MMCV。

**步骤 1.** 安装 MMDetection：请参考[MMDetection的安装文档](https://mmdetection.readthedocs.io/zh_CN/latest/get_started.html#mmdetection)来进行安装。

**步骤 2.** 安装 MMSegmentation：请参考[MMSegmentation的安装文档](https://mmsegmentation.readthedocs.io/zh_CN/latest/get_started.html#id2)来进行安装。

**步骤 3.** 安装MMDetection3d: 请参考[MMDetection3D的安装文档](https://mmdetection3d.readthedocs.io/zh_CN/latest/getting_started.html#id2)来进行安装。



**注意：**MMDetection3d请使用源码编译的方式，下载源码时选择dev分支。



#### 2、准备数据集

（1）下载nuscenes数据集，可以从官网https://nuscenes.org/nuscenes下载数据集压缩包。

由于数据集较大，已经上传到obs://ascend-pytorch-one-datasets/train/nuscenes/ 中，可以参考使用。



（2）数据处理

+ 解压所有压缩包

+ 执行以下命令：

  

在MMDetection3d安装目录下进入"data"目录并放入"nuscenes"数据集，或使用软链接链接到nuscenes数据集，目录结构如下：





#### 3、执行训练

在MMDetection3d安装目录下训练centerpoint模型，输入命令行:

```shell
bash tools/dist_train.sh configs/centerpoint/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus.py 8
```





#### 4、模型结果参考

centerpoint训练参考精度为： 48.54

**注意:**

- 当在NPU上运行centerpoint模型时，需要进行一定修改：

  **修改原因：**

  （1） NPU上的pad算子当前不支持填充int32值，计划在2023年Q2完成修复。

  （2）Voxelization算子存在精度问题，已关联问题单：

  

  **修改方式：** 使用cat算子替换pad算子；Voxelization算子在cpu上计算。
  
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

