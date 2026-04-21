from furniture_bench.furniture.square_table import SquareTable


class OneLeg(SquareTable):
    def __init__(self):
        super().__init__()
        self.should_be_assembled = [(0, 4)]
        self.ignore_z_rot.add((0, 4))
        self.ignore_z_rot_axis[(0, 4)] = 1  # 表示相对旋转中 z 轴是第二维
        self.ori_bound = 0.99  # 忽略 z 轴旋转后，收紧 one_leg 的旋转要求（越接近 1 要求越紧，越接近 0 要求越松）
        """
        0.94 -> 19.95°
        0.95 -> 18.19°
        0.96 -> 16.26°
        0.97 -> 14.07°
        0.98 -> 11.48°
        0.99 -> 8.11°
        0.995 -> 5.73°
        """
        self.assembled_pos_threshold = [0.005, 0.0057, 0.005]  # 放松 one_leg z 轴的位置要求，z 轴是第二维
