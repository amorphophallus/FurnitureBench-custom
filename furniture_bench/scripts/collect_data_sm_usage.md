# collect_data_sm 使用说明

`collect_data_sm.py` 用于通过 SpaceMouse 采集 FurnitureBench 轨迹。启动一次脚本可以连续采集多条轨迹，直到成功轨迹数量达到 `--num-demos`。

## 运行命令

从仓库根目录运行：

```bash
export DATA_DIR_RAW=/path/to/raw-data

python -m furniture_bench.scripts.collect_data_sm \
  --out-data-path "$DATA_DIR_RAW" \
  --furniture one_leg \
  --is-sim \
  --randomness low \
  --num-demos 1 \
  --ctrl-mode diffik \
  --sm-pos-speed 0.6 \
  --sm-rot-speed 6 \
  --teleop-setting 2 \
  --pkl-only
```


## 操作流程

1. 确认 SpaceMouse 驱动和服务已启动。

```bash
sudo systemctl start spacenavd
```

2. 启动采集命令。

3. 看到 `Press s to start` 后，按 `s` 开始当前轨迹。这里使用后台键盘监听，不需要终端窗口获得焦点。等待开始时的 `s` 不会作为移动指令；开始后 `s` 恢复为键盘平移控制。

4. 使用 SpaceMouse 控制末端位姿。两种 setting 都会额外打开 OpenCV 预览窗口显示固定相机和 wrist camera，图像显示为 4 倍大小。`--teleop-setting 1` 使用 world 坐标系控制；`--teleop-setting 2` 使用 end effector 坐标系控制。

5. 使用 SpaceMouse 按钮或键盘 `z` 切换夹爪开合。

6. 当前轨迹成功后，按 `t` 保存为成功轨迹。

7. 当前轨迹失败后，按 `n` 标记失败。默认不保存失败轨迹；加 `--save-failure` 才保存。

8. 脚本保存并 reset 后，再次按 `s` 开始下一条轨迹。

9. 成功轨迹数达到 `--num-demos` 后，脚本结束。

## 常用按键

`t`：标记当前轨迹成功并保存。

`n`：标记当前轨迹失败。只有加 `--save-failure` 时才保存失败轨迹。

`` ` ``：标记一个 skill 完成。

数字键 `0` 到 `9`：手动 reward 标注，用于 `--manual-label` 场景。

`z`：切换夹爪开合；SpaceMouse 按钮也可以切换夹爪。

`w/s/a/d/q/e`：键盘平移控制，可作为 SpaceMouse 外的备用输入。

`i/k/j/l/u/o`：键盘旋转控制，可作为 SpaceMouse 外的备用输入。

`[` 和 `]`：调整键盘控制步长。

## 常用参数


`--out-data-path`：数据保存根目录。脚本会自动在其下创建 `<furniture>` 子目录。

`--furniture`：任务名，例如 `one_leg`、`lamp`、`round_table`、`mug_rack`、`factory_peg_hole`。

`--is-sim`：使用仿真环境。不加则使用真机环境。

`--randomness`：初始化随机程度，可选 `low`、`med`、`high`。

`--num-demos`：目标成功轨迹数量。

`--ctrl-mode`：底层控制器，可选 `osc` 或 `diffik`。

`--gpu-id`：仿真使用的 GPU id，同时用于 compute 和 graphics。

`--pkl-only`：只保存 `.pkl`，不额外保存 mp4/png。采集训练数据时通常建议打开。

`--save-failure`：保存失败轨迹。

`--headless`：无渲染窗口运行。

`--no-ee-laser`：关闭仿真中末端执行器的辅助 laser。

`--sm-pos-speed`：SpaceMouse 平移速度上限，单位是 m/s。默认 `diffik=0.3`，`osc=0.8`。

`--sm-rot-speed`：SpaceMouse 旋转速度上限，单位是 rad/s。默认 `diffik=0.7`，`osc=4.0`。

`--teleop-setting`：SpaceMouse 采集预设，可选 `1` 或 `2`。两种 setting 都会额外显示 `color_image2` 和 `color_image1`，预览尺寸为 4 倍。`1` 平移和旋转都在 world/base 坐标系；`2` 平移和旋转都在 end effector 坐标系，ee 旋转符号为 `[x=+1, y=-1, z=-1]`，同时将 ee 平移的 `dpos[0]` 和 `dpos[2]` 取反。当前 Isaac Gym viewer 没有现成接口把 camera tensor 贴到原生窗口里，所以相机预览使用 OpenCV 窗口。


## 保存路径

脚本会把 `--furniture` 自动拼到 `--out-data-path` 后面：

```text
<out-data-path>/<furniture>/<timestamp>/<timestamp>.pkl
```

例如：

```text
$DATA_DIR_RAW/one_leg/2026-04-29-12-30-00/2026-04-29-12-30-00.pkl
```

如果没有 `--pkl-only`，还会在同一个 timestamp 目录下保存相机视频和深度图。

## 单条 pkl 内容

每个 `.pkl` 是一条 trajectory 的 Python pickle 文件，主要字段包括：

```python
{
    "observations": [...],
    "actions": [...],
    "rewards": [...],
    "skills": [...],
    "success": True or False,
    "furniture": "one_leg",
    "error": False,
    "error_description": "",
}
```

当前入口固定使用 `obs_type="image"`，所以 observation 中通常包含：

```python
{
    "color_image1": ...,  # Wrist cam
    "color_image2": ...,  # Front cam
    "image_size": ...,
    "robot_state": ...,
}
```

真机数据还会额外保存相机内参和相机到 base 的外参。
