"""Define base class for all furniture. It contains the core functions and properties for the furniture (e.g., furniture parts, computing reward function, getting observation,etc.)"""

from abc import ABC
import time
import multiprocessing as mp
from multiprocessing import shared_memory
from typing import Optional, Tuple, List

import numpy as np
import numpy.typing as npt
from gym import logger

import furniture_bench.utils.transform as T
from furniture_bench.utils.pose import (
    is_similar_pos,
    is_similar_rot,
    is_similar_rot_ignore_z,
)
from furniture_bench.config import config
from furniture_bench.furniture.parts.part import Part
from furniture_bench.furniture.parts.obstacle_front import ObstacleFront
from furniture_bench.furniture.parts.obstacle_right import ObstacleRight
from furniture_bench.furniture.parts.obstacle_left import ObstacleLeft

from ipdb import set_trace as bp


class Furniture(ABC):
    def __init__(self):
        self.parts: List[Part] = []
        self.num_parts = len(self.parts)

        self.ori_bound = 0.94
        # Multi processing for ,camera, pose detection.
        self.detection_started = False
        # Relative pose for coordinate base tag from robot base frame.
        self.color_shape = (
            config["camera"]["color_img_size"][1],
            config["camera"]["color_img_size"][0],
            3,
        )
        self.depth_shape = (
            config["camera"]["depth_img_size"][1],
            config["camera"]["depth_img_size"][0],
        )
        self.robot_pos_lim = config["robot"]["position_limits"]
        self.parts_pos_lim = config["furniture"]["position_limits"]

        # Defined in the child class.
        self.reset_temporal_xys = None
        self.reset_temporal_idxs = {}
        self.should_assembled_first = {}
        self.should_be_assembled = []
        self.assembled_rel_poses = {}

        # Reset assembled set.
        self.assembled_set = set()
        self.position_only = set()
        self.ignore_z_rot = set()
        self.ignore_z_rot_axis = {}
        self.assembly_debug = False
        self.max_env_steps = 3000

        self._init_obstacle()

        self.reset_pos_diff_threshold = [0.015, 0.015, 0.015]  # 1.5cm.
        self.reset_ori_bound = 0.96  # 15 degrees.
        self.max_env_steps_skills = [0, 250, 250, 250, 250, 350]
        self.max_env_steps_from_skills = [
            sum(self.max_env_steps_skills[i:])
            for i in range(len(self.max_env_steps_skills) - 1)
        ]
        # Check whether the furniture is assembled.
        # If the value is smaller than these thresholds, we consider the furniture is assembled.
        self.assembled_pos_threshold = config["furniture"]["assembled_pos_threshold"]

    def randomize_init_pose(
        self, from_skill, pos_range=[-0.05, 0.05], rot_range=45
    ) -> bool:
        """Randomize the furniture initial pose."""
        trial = 0
        max_trial = 300000
        while True:
            trial += 1
            for part in self.parts:
                part.randomize_init_pose(from_skill, pos_range, rot_range)
            if trial > max_trial:
                logger.error("Failed to randomize init pose")
                return False
            if self._in_boundary(from_skill) and not self._check_collision():
                # logger.info("Found collision-free init pose")
                return True

    def randomize_high(self, high_random_idx: int):
        """Initialize furniture parts with predefined poses of high randomness."""
        for part in self.parts:
            part.randomize_init_pose_high(high_random_idx)

    def randomize_skill_init_pose(self, from_skill) -> bool:
        """Randomize the furniture initial pose."""
        trial = 0
        max_trial = 300000
        while True:
            trial += 1
            for i, part in enumerate(self.parts):
                if part.part_moved_skill_idx <= from_skill:
                    # Reduce randomized range the part that has been moved from the skill.
                    part.randomize_init_pose(
                        from_skill=from_skill,
                        pos_range=[-0.0, 0.0],
                        rot_range=0,
                    )
                elif (
                    part.part_attached_skill_idx <= from_skill
                    and self.skill_attach_part_idx == i
                ):
                    attached_part, attach_to = self.attach(part)
                    if attached_part:
                        self.set_attached_pose(part, attach_to, from_skill)
                else:
                    part.randomize_init_pose(from_skill=from_skill)
            if trial > max_trial:
                logger.error("Failed to randomize init pose")
                return False
            if not self._check_collision(from_skill) and self._in_boundary(from_skill):
                # logger.info("Found initialization pose")
                return True

    def _check_collision(self):
        """Simple rectangle collision check between two parts."""
        for i in range(self.num_parts):
            for j in range(i + 1, self.num_parts):
                if self.parts[i].is_collision(self.parts[j]):
                    return True

        for i in range(self.num_parts):
            for obstacle in self.obstacles:
                if self.parts[i].is_collision(obstacle):
                    return True

        return False

    def _in_boundary(self, from_skill):
        """Check whether the furniture is in the boundary."""
        for part in self.parts:
            if not part.in_boundary(self.parts_pos_lim, from_skill):
                return False
        return True

    def check_parts_in_reset_pose(self, from_skill, check_found_only=False) -> bool:
        parts_poses, founds = self.get_parts_poses_founds()
        for part in self.parts:
            part_idx = part.part_idx
            part_pose = parts_poses[part_idx * 7 : (part_idx + 1) * 7]
            part_pose = T.pose2mat(part_pose)
            if not founds[part_idx]:
                print(
                    f"[reset] Part {self.__class__.__name__} [{part_idx}] is not found"
                )

                if check_found_only:
                    return False

            if not check_found_only and (
                not founds[part_idx]
                or not part.is_in_reset_pose(
                    part_pose,
                    from_skill,
                    self.reset_pos_diff_threshold,
                    self.reset_ori_bound,
                )
            ):
                return False
        return True

    def randomize_high(self, high_random_idx: int):
        """Initialize furniture parts with predefined poses of high randomness."""
        for part in self.parts:
            part.randomize_init_pose_high(high_random_idx)

    def reset_pose_filter(self):
        for part in self.parts:
            part.reset_pose_filters()

    def get_parts_poses(self):
        if not self.detection_started:
            raise Exception("First call `start_detection` to get part poses")
        with self.lock:
            return self.get_array()

    def get_parts_poses_founds(
        self,
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        """Get parts poses and founds only."""
        parts_poses, founds, _, _, _, _, _, _ = self.get_parts_poses()
        return parts_poses, founds

    def get_front_image(self) -> npt.NDArray[np.uint8]:
        """Get front image of the furniture only."""
        _, _, color_image1, _, _, _, _, _ = self.get_parts_poses()
        return color_image1

    def get_part_pose(self, part_idx):
        max_trial = 5
        for _ in range(max_trial):
            parts_poses, _, _, _, _, _, _, _ = self.get_parts_poses()
            part_pose = parts_poses[part_idx * 7 : (1 + part_idx) * 7]
            if np.isclose(part_pose, np.zeros((7,))).all():
                time.sleep(0.2)
                continue
            return T.pose2mat(part_pose)

    def start_detection(self):
        # Import the RealSense-backed detector only when live detection is
        # requested.  Offline consumers (for example pickle annotation) only
        # need the furniture geometry and should not require camera hardware
        # merely to instantiate a Furniture object.
        from furniture_bench.utils.detection import detection_loop

        self.ctx = mp.get_context("spawn")

        if self.detection_started:
            return

        self.shm = self.create_shared_memory()
        self.lock = self.ctx.Lock()

        self.proc = self.ctx.Process(
            target=detection_loop,
            args=(
                config,
                self.parts,
                self.num_parts,
                self.tag_size,
                self.lock,
                self.shm,
            ),
            daemon=True,
        )
        self.proc.start()
        self.detection_started = True
        self._wait_detection_start()

    def _wait_detection_start(self):
        max_wait = 20  # 20 seconds
        while True:
            start = time.time()
            while (time.time() - start) < max_wait:
                _, founds = self.get_parts_poses_founds()
                if founds.any():
                    # Heuristic to check whether the detection started. (At least one part is found.)
                    return
                time.sleep(0.03)

            input(
                "Could not find any furniture parts from the cameras\n Press enter after putting the furniture in the workspace."
            )

    def get_array(
        self,
    ) -> Tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.bool_],
        npt.NDArray[np.uint8],
        npt.NDArray[np.uint16],
        npt.NDArray[np.uint8],
        npt.NDArray[np.uint16],
        npt.NDArray[np.uint8],
        npt.NDArray[np.uint16],
    ]:
        """Get the shared memory of parts poses and images."""
        parts_poses_shm = shared_memory.SharedMemory(name=self.shm[0])
        parts_founds_shm = shared_memory.SharedMemory(name=self.shm[1])
        color_shm1 = shared_memory.SharedMemory(name=self.shm[2])
        depth_shm1 = shared_memory.SharedMemory(name=self.shm[3])
        color_shm2 = shared_memory.SharedMemory(name=self.shm[4])
        depth_shm2 = shared_memory.SharedMemory(name=self.shm[5])
        color_shm3 = shared_memory.SharedMemory(name=self.shm[6])
        depth_shm3 = shared_memory.SharedMemory(name=self.shm[7])

        parts_poses = np.ndarray(
            shape=(self.num_parts * 7,), dtype=np.float32, buffer=parts_poses_shm.buf
        )
        parts_found = np.ndarray(
            shape=(self.num_parts,), dtype=bool, buffer=parts_founds_shm.buf
        )
        color_img1 = np.ndarray(
            shape=self.color_shape, dtype=np.uint8, buffer=color_shm1.buf
        )
        depth_img1 = np.ndarray(
            shape=self.depth_shape, dtype=np.uint16, buffer=depth_shm1.buf
        )
        color_img2 = np.ndarray(
            shape=self.color_shape, dtype=np.uint8, buffer=color_shm2.buf
        )
        depth_img2 = np.ndarray(
            shape=self.depth_shape, dtype=np.uint16, buffer=depth_shm2.buf
        )
        color_img3 = np.ndarray(
            shape=self.color_shape, dtype=np.uint8, buffer=color_shm3.buf
        )
        depth_img3 = np.ndarray(
            shape=self.depth_shape, dtype=np.uint16, buffer=depth_shm3.buf
        )

        return (
            parts_poses.copy(),
            parts_found.copy(),
            color_img1.copy(),
            depth_img1.copy(),
            color_img2.copy(),
            depth_img2.copy(),
            color_img3.copy(),
            depth_img3.copy(),
        )

    def create_shared_memory(self) -> Tuple[str, str, str, str, str, str, str, str]:
        """Create shared memory to save the parts poses and images."""
        parts_poses = np.zeros(shape=(self.num_parts * 7,), dtype=np.float32)
        parts_poses_shm = shared_memory.SharedMemory(
            create=True, size=parts_poses.nbytes
        )

        parts_founds = np.zeros(shape=(self.num_parts,), dtype=bool)
        parts_founds_shm = shared_memory.SharedMemory(
            create=True, size=parts_founds.nbytes
        )

        color_shm1, depth_shm1 = self._create_shared_memory_for_img()
        color_shm2, depth_shm2 = self._create_shared_memory_for_img()
        color_shm3, depth_shm3 = self._create_shared_memory_for_img()

        return (
            parts_poses_shm.name,
            parts_founds_shm.name,
            color_shm1.name,
            depth_shm1.name,
            color_shm2.name,
            depth_shm2.name,
            color_shm3.name,
            depth_shm3.name,
        )

    def _create_shared_memory_for_img(self):
        """Utility to create shared memories for images."""
        color_img = np.zeros(shape=self.color_shape, dtype=np.uint8)
        color_shm = shared_memory.SharedMemory(create=True, size=color_img.nbytes)
        depth_img = np.zeros(shape=self.depth_shape, dtype=np.uint16)
        depth_shm = shared_memory.SharedMemory(create=True, size=depth_img.nbytes)

        return color_shm, depth_shm

    def reset(self):
        """Reset filter and assembled parts."""
        self.reset_pose_filter()
        self.assembled_set = set()
        for part in self.parts:
            part.reset()

    def compute_assemble(
        self,
        parts_poses: Optional[npt.NDArray[np.float32]] = None,
        founds: Optional[npt.NDArray[np.bool_]] = None,
    ) -> int:
        """Update the set of assembled parts and return the number of newly assembled parts."""
        ret = 0
        for assemble_idx in self.should_be_assembled:
            part_idx1, part_idx2 = assemble_idx
            pair = (part_idx1, part_idx2)
            if self.is_assembled_idx(part_idx1, part_idx2, parts_poses, founds):
                if pair not in self.assembled_set:
                    logger.info(
                        f"{self.parts[pair[0]].name} (id: {pair[0]}), {self.parts[pair[1]].name} (id: {pair[1]}) are assembled."
                    )
                    self.assembled_set.add(pair)
                    self._log_assemble_set()
                    ret += 1

        return ret

    def _log_assemble_set(self):
        log_messages = ["Assembled Set"]
        for i, assembled in enumerate(self.assembled_set):
            log_message = f"[{self.parts[assembled[0]].name} (id: {assembled[0]}), {self.parts[assembled[1]].name} (id: {assembled[1]})]"
            # Not last element of the set
            if not i == len(self.assembled_set) - 1:
                log_message += " / "
            log_messages.append(log_message)
        full_log_message = " ".join(log_messages)
        logger.info(full_log_message)

    def manual_assemble_label(self, part_idx):
        """Manually label assembled with keyboard input."""
        for assemble_idx in self.should_be_assembled:
            part_idx1, part_idx2 = assemble_idx
            pair = (part_idx1, part_idx2)
            if (
                part_idx == assemble_idx[0] or part_idx == assemble_idx[1]
            ) and pair not in self.assembled_set:
                self._log_assemble_set()
                self.assembled_set.add(pair)
                logger.info(f"{pair} assembled")
                return 1
        return 0

    def all_assembled(self) -> bool:
        if len(self.assembled_set) == len(self.should_be_assembled):
            return True
        return False

    def parts_out_pos_lim(self) -> bool:
        parts_poses, _ = self.get_parts_poses_founds()
        for part_idx in range(len(self.parts)):
            part_pose = parts_poses[7 * part_idx : 7 * (part_idx + 1)]
            if not self.is_in_pos_lim(part_pose):
                logger.warn(
                    f"[env] part {self.parts[part_idx]} {[part_idx]} out of positional limits."
                )
                return True
        return False

    def is_in_pos_lim(self, part_pose: npt.NDArray[np.float32]) -> bool:
        """Test whether the part_pose is in robot's pos limit.
        We only checks the maximum height of Z since detection of z is sometimes negative because of the detection error.
        """
        part_pose = config["robot"]["tag_base_from_robot_base"] @ T.pose2mat(part_pose)
        part_pos = part_pose[:3, 3]
        return (
            part_pos[0] > self.robot_pos_lim[0][0]
            and part_pos[0] < self.robot_pos_lim[0][1]
            and part_pos[1] > self.robot_pos_lim[1][0]
            and part_pos[1] < self.robot_pos_lim[1][1]
            and part_pos[2] < self.robot_pos_lim[2][1]
        )

    def _assembly_rot_debug_metrics(
        self,
        current_rot: npt.NDArray[np.float32],
        target_rot: npt.NDArray[np.float32],
    ):
        rel_rot = current_rot @ target_rot.T
        trace = np.trace(rel_rot)
        cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        full_angle_rad = float(np.arccos(cos_angle))

        if np.isclose(full_angle_rad, 0.0):
            axis_angle = np.zeros(3, dtype=current_rot.dtype)
        else:
            axis = np.array(
                [
                    rel_rot[2, 1] - rel_rot[1, 2],
                    rel_rot[0, 2] - rel_rot[2, 0],
                    rel_rot[1, 0] - rel_rot[0, 1],
                ],
                dtype=current_rot.dtype,
            ) / (2.0 * np.sin(full_angle_rad))
            axis_angle = axis * full_angle_rad

        ignore_z_axis_angle = axis_angle.copy()
        ignore_z_axis_angle[1] = 0.0
        ignore_z_angle_rad = float(np.linalg.norm(ignore_z_axis_angle))

        column_cosines = []
        for col_idx in range(3):
            current_col = current_rot[:, col_idx]
            target_col = target_rot[:, col_idx]
            column_cosines.append(
                float(
                    np.dot(current_col, target_col)
                    / (np.linalg.norm(current_col) * np.linalg.norm(target_col))
                )
            )

        return (
            full_angle_rad,
            ignore_z_angle_rad,
            axis_angle.tolist(),
            ignore_z_axis_angle.tolist(),
            column_cosines,
        )

    def _format_debug_value(self, value):
        if isinstance(value, np.ndarray):
            return self._format_debug_value(value.tolist())
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(self._format_debug_value(v) for v in value) + "]"
        if isinstance(value, (np.bool_, bool)):
            return str(bool(value))
        if isinstance(value, (np.integer, int)):
            return str(int(value))
        if isinstance(value, (np.floating, float)):
            return f"{float(value):.2f}"
        return str(value)

    def get_ignore_z_rot_axis(self, pair: Optional[Tuple[int, int]]) -> int:
        if pair is None:
            return 1
        return int(self.ignore_z_rot_axis.get(pair, 1))

    def _print_assembly_debug(
        self,
        pair: Optional[Tuple[int, int]],
        rel_pose,
        assembled_rel_pose,
        pose_idx: int,
        similar_rot: bool,
        similar_pos: bool,
        similar_pose: bool,
        source: str,
    ):
        if not self.assembly_debug or pair is None:
            return

        part1_name = self.parts[pair[0]].name
        part2_name = self.parts[pair[1]].name
        current_pos = rel_pose[:3, 3]
        target_pos = assembled_rel_pose[:3, 3]
        pos_diff = current_pos - target_pos
        abs_pos_diff = np.abs(pos_diff)
        pos_threshold = np.asarray(self.assembled_pos_threshold, dtype=np.float32)
        pos_within_threshold = (abs_pos_diff <= pos_threshold).tolist()

        current_rot = rel_pose[:3, :3]
        target_rot = assembled_rel_pose[:3, :3]
        (
            full_angle_rad,
            ignore_z_angle_rad,
            axis_angle,
            ignore_z_axis_angle,
            column_cosines,
        ) = self._assembly_rot_debug_metrics(current_rot, target_rot)

        use_position_only = pair in self.position_only
        use_ignore_z_rot = pair in self.ignore_z_rot
        ignore_z_rot_axis = self.get_ignore_z_rot_axis(pair)
        if use_ignore_z_rot:
            ignore_z_axis_angle[0] = axis_angle[0]
            ignore_z_axis_angle[1] = axis_angle[1]
            ignore_z_axis_angle[2] = axis_angle[2]
            ignore_z_axis_angle[ignore_z_rot_axis] = 0.0
            ignore_z_angle_rad = float(np.linalg.norm(ignore_z_axis_angle))
        rot_threshold_rad = (
            None
            if use_position_only
            else float(np.arccos(np.clip(self.ori_bound, -1.0, 1.0)))
        )

        print(
            "[assembly_debug] "
            f"furniture={self.name} "
            f"source={source} "
            f"pair=({part1_name},{part2_name}) "
            f"target_pose_idx={pose_idx} "
            f"mode={'position_only' if use_position_only else 'ignore_z_rot' if use_ignore_z_rot else 'full_rot'} "
            f"ignore_axis={ignore_z_rot_axis if use_ignore_z_rot else 'n/a'} "
            f"similar_rot={similar_rot} "
            f"similar_pos={similar_pos} "
            f"assembled={similar_pose}"
        )
        print(f"  current_rel_pos={self._format_debug_value(current_pos)}")
        print(f"  target_rel_pos={self._format_debug_value(target_pos)}")
        print(f"  rel_pos_diff={self._format_debug_value(pos_diff)}")
        print(f"  abs_rel_pos_diff={self._format_debug_value(abs_pos_diff)}")
        print(f"  pos_threshold={self._format_debug_value(pos_threshold)}")
        print(f"  pos_within_threshold={self._format_debug_value(pos_within_threshold)}")
        print(f"  current_rel_rot={self._format_debug_value(current_rot)}")
        print(f"  target_rel_rot={self._format_debug_value(target_rot)}")
        print(f"  rot_column_cosines={self._format_debug_value(column_cosines)}")
        print(
            f"  rot_full_angle_rad={self._format_debug_value(full_angle_rad)} "
            f"rot_full_angle_deg={self._format_debug_value(np.degrees(full_angle_rad))}"
        )
        print(
            "  rot_ignore_z_angle_rad="
            f"{self._format_debug_value(ignore_z_angle_rad)} "
            f"rot_ignore_z_angle_deg={self._format_debug_value(np.degrees(ignore_z_angle_rad))}"
        )
        print(f"  rel_axis_angle={self._format_debug_value(axis_angle)}")
        print(
            "  rel_axis_angle_ignore_z="
            f"{self._format_debug_value(ignore_z_axis_angle)}"
        )
        if rot_threshold_rad is None:
            print("  rot_threshold=position_only")
        else:
            print(
                "  rot_threshold_rad="
                f"{self._format_debug_value(rot_threshold_rad)} "
                f"rot_threshold_deg={self._format_debug_value(np.degrees(rot_threshold_rad))} "
                f"ori_bound={self._format_debug_value(self.ori_bound)}"
            )

    def is_assembled_idx(
        self,
        part_idx1: int,
        part_idx2: int,
        parts_poses: Optional[npt.NDArray[np.float32]] = None,
        founds: Optional[npt.NDArray[np.bool_]] = None,
    ) -> bool:
        """Compute whether the part_idx1 and part_idx2 are assembled or not."""
        if (part_idx1, part_idx2) not in self.should_be_assembled:
            return False

        if parts_poses is None:
            parts_poses, founds = self.get_parts_poses_founds()

        pose1 = parts_poses[7 * part_idx1 : 7 * (part_idx1 + 1)]
        pose2 = parts_poses[7 * part_idx2 : 7 * (part_idx2 + 1)]

        # The part not found.
        if not founds[part_idx1] or not founds[part_idx2]:
            if (part_idx1, part_idx2) in self.assembled_set:
                return True
            return False

        if not self.check_assembled_first(part_idx1, part_idx2):
            return False

        pose1_mat = T.pose2mat(pose1)
        pose2_mat = T.pose2mat(pose2)
        rel_pose = np.linalg.inv(pose1_mat) @ pose2_mat

        assembled_rel_poses = self.assembled_rel_poses[(part_idx1, part_idx2)]
        if assembled_rel_poses is None:
            raise Exception("No relative pose!")

        pair = (part_idx1, part_idx2)
        for pose_idx, assembled_rel_pose in enumerate(assembled_rel_poses):
            if pair in self.position_only:
                similar_rot = True
            elif pair in self.ignore_z_rot:
                similar_rot = is_similar_rot_ignore_z(
                    rel_pose[:3, :3],
                    assembled_rel_pose[:3, :3],
                    self.ori_bound,
                    ignore_z_rot_axis=self.get_ignore_z_rot_axis(pair),
                )
            else:
                similar_rot = is_similar_rot(
                    rel_pose[:3, :3],
                    assembled_rel_pose[:3, :3],
                    self.ori_bound,
                )
            similar_pos = is_similar_pos(
                rel_pose[:3, 3],
                assembled_rel_pose[:3, 3],
                self.assembled_pos_threshold,
            )
            similar_pose = similar_rot and similar_pos
            self._print_assembly_debug(
                pair=pair,
                rel_pose=rel_pose,
                assembled_rel_pose=assembled_rel_pose,
                pose_idx=pose_idx,
                similar_rot=similar_rot,
                similar_pos=similar_pos,
                similar_pose=similar_pose,
                source="is_assembled_idx",
            )
            if similar_pose:
                return True

        return False

    def assembled(
        self,
        rel_pose,
        assembled_rel_poses,
        pair: Optional[Tuple[int, int]] = None,
    ):
        for pose_idx, assembled_rel_pose in enumerate(assembled_rel_poses):
            if pair is not None and pair in self.position_only:
                similar_rot = True
            elif pair is not None and pair in self.ignore_z_rot:
                similar_rot = is_similar_rot_ignore_z(
                    rel_pose[:3, :3],
                    assembled_rel_pose[:3, :3],
                    self.ori_bound,
                    ignore_z_rot_axis=self.get_ignore_z_rot_axis(pair),
                )
            else:
                similar_rot = is_similar_rot(
                    rel_pose[:3, :3],
                    assembled_rel_pose[:3, :3],
                    self.ori_bound,
                )
            similar_pos = is_similar_pos(
                rel_pose[:3, 3],
                assembled_rel_pose[:3, 3],
                self.assembled_pos_threshold,
            )
            similar_pose = similar_rot and similar_pos
            self._print_assembly_debug(
                pair=pair,
                rel_pose=rel_pose,
                assembled_rel_pose=assembled_rel_pose,
                pose_idx=pose_idx,
                similar_rot=similar_rot,
                similar_pos=similar_pos,
                similar_pose=similar_pose,
                source="assembled",
            )
            if similar_pose:
                return True

        return False

    def _init_obstacle(self):
        """Initialize the obstacle."""
        self.obstacles = [ObstacleFront(), ObstacleLeft(), ObstacleRight()]

    def check_assembled_first(self, part_idx1: int, part_idx2: int) -> bool:
        """Check if the parts that should be assembled before (part_idx1, part_idx2) are assembled."""
        if self.should_assembled_first.get((part_idx1, part_idx2)) is not None:
            idx1, idx2 = self.should_assembled_first[(part_idx1, part_idx2)]
            if (idx1, idx2) not in self.assembled_set:
                return False

        return True

    def __del__(self):
        """Clean the resources."""
        if self.detection_started:
            for name in self.shm:
                m = shared_memory.SharedMemory(name=name, create=False)
                m.close()
                m.unlink()
            self.proc.terminate()
