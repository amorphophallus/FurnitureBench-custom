import numpy as np
import os
import torch

from furniture_bench.utils.pose import get_mat, rot_mat
from furniture_bench.utils.pose import cosine_sim
from furniture_bench.furniture.parts.leg import Leg
import furniture_bench.controllers.control_utils as C
from furniture_bench.config import config


class RoundTableLeg(Leg):
    _gripper_width_key = "round_table"

    def __init__(self, part_config, part_idx):
        self.reset_x_len = 0.0425
        self.reset_y_len = 0.09125

        self.tag_offset = 0.01625
        self.half_width = 0.01625
        super().__init__(part_config, part_idx)

        self.reset_gripper_width = 0.05
        self.grasp_margin_x = 0.020
        self.grasp_margin_z = 0.03
        self.insert_height = 0.054

    def is_in_reset_ori(self, pose, from_skill, ori_bound):
        reset_ori = (
            self.reset_ori[from_skill] if len(self.reset_ori) > 1 else self.reset_ori[0]
        )
        return cosine_sim(reset_ori[:3, 1], pose[:3, 1]) > 0.9

    def reset(self):
        self.prev_pose = None
        self._state = "reach_leg_floor_xy"
        self.gripper_action = -1
        self.reset_skill_state()

    def reset_skill_state(self):
        super().reset_skill_state()
        self.skill_place_xy_threshold = 0.008
        self.skill_place_z_threshold = 0.05
        self.skill_place_ori_threshold = 0.5

    def _compute_skill_pick_target(
        self,
        ee_pose_robot,
        rb_states,
        part_idxs,
        sim_to_april_mat,
        april_to_robot,
    ):
        leg_pose_env = C.to_homogeneous(
            rb_states[part_idxs[self.name]][0][:3],
            C.quat2mat(rb_states[part_idxs[self.name]][0][3:7]),
        )
        leg_pose_april = sim_to_april_mat @ leg_pose_env
        leg_pose_april = self._find_down_z(leg_pose_april).clone().to(ee_pose_robot.device)
        leg_pose_april = (
            torch.tensor(get_mat([0, 0.043, 0], [0, 0, 0]), device=ee_pose_robot.device)
            @ leg_pose_april
        )
        pos_april = leg_pose_april[:4, 3]
        target_pos_robot = (april_to_robot @ pos_april)[:3]
        target_pos_robot[2] = ee_pose_robot[2, 3]
        return target_pos_robot

    def _compute_skill_pick_pose_target(
        self,
        ee_pose_robot,
        rb_states,
        part_idxs,
        sim_to_april_mat,
        april_to_robot,
    ):
        device = ee_pose_robot.device
        leg_pose_env = C.to_homogeneous(
            rb_states[part_idxs[self.name]][0][:3],
            C.quat2mat(rb_states[part_idxs[self.name]][0][3:7]),
        )
        leg_pose_april = sim_to_april_mat @ leg_pose_env
        leg_pose_april = self._find_down_z(leg_pose_april).clone().to(device)
        target_pos_robot = super()._compute_skill_pick_target(
            ee_pose_robot,
            rb_states,
            part_idxs,
            sim_to_april_mat,
            april_to_robot,
        )
        margin = torch.tensor(rot_mat([0, -np.pi / 5, 0], hom=True), device=device).float()
        grasp_rot = torch.tensor(
            rot_mat([np.pi / 2, -np.pi / 2, 0], hom=True), device=device
        ).float()
        theta_y = torch.acos(
            torch.clamp(leg_pose_april[1, 1], min=-1.0, max=1.0)
        ).detach().cpu().numpy()
        sign = 1 if leg_pose_april[0, 1] > 0 else -1
        target_ori_robot = (
            torch.tensor(rot_mat([0, 0, sign * theta_y], hom=True), device=device).float()
            @ margin
            @ april_to_robot
            @ grasp_rot
        )[:3, :3]
        return self._guidance_pose_from_point(target_pos_robot, target_ori_robot)

    def _compute_skill_place_guidance_orientation(self, device):
        return torch.tensor(
            get_mat([0, 0, 0], [np.pi / 2, 0, -np.pi / 4]),
            device=device,
        ).float()[:3, :3]

    def _find_leg_pose_x_look_front_skill(self, leg_pose, device):
        best_leg_pose = leg_pose.clone()
        tmp_leg_pose = leg_pose
        rot = torch.tensor(rot_mat([0, -np.pi / 2, 0], hom=True), device=device).float()
        for _ in range(3):
            tmp_leg_pose = tmp_leg_pose @ rot
            if best_leg_pose[0, 0] < tmp_leg_pose[0, 0]:
                best_leg_pose = tmp_leg_pose
        return best_leg_pose

    def _compute_skill_place_target(
        self,
        ee_pose_robot,
        rb_states,
        part_idxs,
        sim_to_april_mat,
        april_to_robot,
        assemble_to,
    ):
        device = ee_pose_robot.device
        base_pose_env = C.to_homogeneous(
            rb_states[part_idxs[assemble_to]][0][:3],
            C.quat2mat(rb_states[part_idxs[assemble_to]][0][3:7]),
        )
        leg_pose_env = C.to_homogeneous(
            rb_states[part_idxs[self.name]][0][:3],
            C.quat2mat(rb_states[part_idxs[self.name]][0][3:7]),
        )
        base_pose_april = sim_to_april_mat @ base_pose_env
        leg_pose_april = sim_to_april_mat @ leg_pose_env
        leg_pose_robot = april_to_robot @ leg_pose_april
        base_pose_robot = april_to_robot @ base_pose_april
        target_leg_pose_robot = (
            base_pose_robot
            @ torch.tensor(self.default_assembled_pose, device=device).float()
        )
        rel_robot = target_leg_pose_robot @ torch.linalg.inv(leg_pose_robot)
        target_ee_pose_robot = rel_robot @ ee_pose_robot
        self.skill_target_part_pose_robot = target_leg_pose_robot.clone()
        self.skill_target_anchor_pose_robot = base_pose_robot.clone()
        return target_ee_pose_robot

    def _compute_skill_insert_target(
        self,
        ee_pose_robot,
        rb_states,
        part_idxs,
        sim_to_april_mat,
        april_to_robot,
        assemble_to,
    ):
        return self._compute_skill_place_target(
            ee_pose_robot,
            rb_states,
            part_idxs,
            sim_to_april_mat,
            april_to_robot,
            assemble_to,
        )

    def _compute_skill_screw_target(
        self,
        rb_states,
        part_idxs,
        sim_to_april_mat,
        april_to_robot,
        assemble_to,
    ):
        leg_pose_env = C.to_homogeneous(
            rb_states[part_idxs[self.name]][0][:3],
            C.quat2mat(rb_states[part_idxs[self.name]][0][3:7]),
        )
        leg_pose_april = sim_to_april_mat @ leg_pose_env
        leg_pose_robot = april_to_robot @ leg_pose_april
        long_axis_idx = 1 if self.reset_y_len >= self.reset_x_len else 0
        long_axis_len = self.reset_y_len if long_axis_idx == 1 else self.reset_x_len
        long_axis_robot = leg_pose_robot[:3, long_axis_idx]
        long_axis_offset_robot = long_axis_robot * (long_axis_len * 0.25)
        return (leg_pose_robot[:3, 3] + long_axis_offset_robot).clone()

    @staticmethod
    def _point_to_segment_distance(point_env, start_env, end_env):
        segment_env = end_env - start_env
        denom = torch.dot(segment_env, segment_env).clamp_min(1e-8)
        t = torch.dot(point_env - start_env, segment_env) / denom
        t = torch.clamp(t, 0.0, 1.0)
        closest_env = start_env + t * segment_env
        return torch.linalg.norm(point_env - closest_env)

    def _finger_leg_capsule_distances_env(self, left_finger_pos, right_finger_pos, rb_states, part_idxs):
        leg_pose_env = self._part_pose_env(self.name, rb_states, part_idxs)
        leg_axis_env = leg_pose_env[:3, 1]
        leg_axis_env = leg_axis_env / torch.linalg.norm(leg_axis_env).clamp_min(1e-8)
        half_len = self.reset_y_len * 0.5
        leg_center_env = leg_pose_env[:3, 3]
        segment_start_env = leg_center_env - leg_axis_env * half_len
        segment_end_env = leg_center_env + leg_axis_env * half_len
        left_dist = self._point_to_segment_distance(
            left_finger_pos, segment_start_env, segment_end_env
        )
        right_dist = self._point_to_segment_distance(
            right_finger_pos, segment_start_env, segment_end_env
        )
        return left_dist, right_dist

    @staticmethod
    def _debug_scalar(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            return float(value.detach().cpu().item())
        return float(value)

    def _is_seated(
        self,
        rb_states,
        part_idxs,
        assemble_to,
        sim_to_april_mat,
        april_to_robot,
    ):
        """Return True when the leg is roughly at the place target (near the hole).

        Uses the same part-relative error as the place->insert transition but
        with 2x looser thresholds and ignore_axis=2 (matching the place phase
        which uses ignore_axis=2 for round table).
        """
        if (
            self.skill_target_part_pose_robot is None
            or self.skill_target_anchor_pose_robot is None
        ):
            return True  # targets not established yet; don't claim dropped
        leg_pose_robot = self._part_pose_robot(
            self.name, rb_states, part_idxs, sim_to_april_mat, april_to_robot
        )
        base_pose_robot = self._part_pose_robot(
            assemble_to, rb_states, part_idxs, sim_to_april_mat, april_to_robot
        )
        xy_error, z_error, ori_error = self._part_place_errors_robot(
            leg_pose_robot, base_pose_robot, ignore_axis=2
        )
        seated = (
            xy_error < self.skill_place_part_xz_threshold * 2.0
            and z_error < self.skill_place_z_threshold * 2.0
            and ori_error < self.skill_place_part_ori_threshold * 2.0
        )
        return bool(seated)

    def update_skill_state(
        self,
        ee_pos,
        ee_quat,
        gripper_width,
        rb_states,
        part_idxs,
        sim_to_april_mat,
        april_to_robot,
        left_finger_pos,
        right_finger_pos,
        left_finger_force,
        right_finger_force,
        part_force,
        assemble_to,
        assembled=False,
    ):
        ee_pose_robot = C.to_homogeneous(ee_pos, C.quat2mat(ee_quat))
        table_normal = torch.tensor([0.0, 0.0, 1.0], device=ee_pos.device)
        grasp_axis = right_finger_pos - left_finger_pos
        grasp_axis[2] = 0.0
        if torch.linalg.norm(grasp_axis) < 1e-6:
            grasp_axis = torch.tensor([1.0, 0.0, 0.0], device=ee_pos.device)

        # Reverse edges place/insert/screw -> pick: if the leg was released
        # during assembly (dropped) and is no longer seated at the hole,
        # restart from pick.
        if self.skill_state in ("place", "insert", "screw"):
            held = self._is_held(
                gripper_width,
                part_force,
                left_finger_pos,
                right_finger_pos,
                rb_states,
                part_idxs,
                table_normal,
            )
            if not held and not self._is_seated(
                rb_states,
                part_idxs,
                assemble_to,
                sim_to_april_mat,
                april_to_robot,
            ):
                self.reset_skill_state()
                return self.skill_state

        if self.skill_state == "pick":
            self.skill_guidance_point_robot = super()._compute_skill_pick_target(
                ee_pose_robot,
                rb_states,
                part_idxs,
                sim_to_april_mat,
                april_to_robot,
            )
            self.skill_guidance_pose_robot = self._compute_skill_pick_pose_target(
                ee_pose_robot,
                rb_states,
                part_idxs,
                sim_to_april_mat,
                april_to_robot,
            )
            self.skill_target_gripper_width = self.half_width * 2
            _ = self._compute_skill_place_target(
                ee_pose_robot,
                rb_states,
                part_idxs,
                sim_to_april_mat,
                april_to_robot,
                assemble_to,
            )
            base_pose_robot = self._part_pose_robot(
                assemble_to, rb_states, part_idxs, sim_to_april_mat, april_to_robot
            )
            leg_pose_robot = self._part_pose_robot(
                self.name, rb_states, part_idxs, sim_to_april_mat, april_to_robot
            )
            xy_error, z_error, ori_error = self._part_place_errors_robot(
                leg_pose_robot,
                base_pose_robot,
                ignore_axis=2,
            )
            place_ok = (
                xy_error < self.skill_place_part_xz_threshold
                and z_error < self.skill_place_z_threshold
                and ori_error < self.skill_place_part_ori_threshold
            )
            if place_ok:
                self.skill_state = "insert"
                self.skill_target_ee_pose_robot = self._compute_skill_insert_target(
                    ee_pose_robot,
                    rb_states,
                    part_idxs,
                    sim_to_april_mat,
                    april_to_robot,
                    assemble_to,
                )
                self.skill_guidance_point_robot = (
                    self.skill_target_ee_pose_robot[:3, 3].clone()
                )
                self.skill_guidance_pose_robot = self._guidance_pose_from_point(
                    self.skill_guidance_point_robot,
                    self._compute_skill_place_guidance_orientation(ee_pose_robot.device),
                )
                self.skill_target_gripper_width = self.half_width * 2
                return self.skill_state
            pinched = False
            leg_force_mag = None
            left_dist = None
            right_dist = None
            close_to_leg = False
            left_capsule_dist = None
            right_capsule_dist = None
            capsule_close_to_leg = False
            narrow_gripper = (
                gripper_width
                < config["robot"]["max_gripper_width"]["round_table"] * 0.8
            )
            if part_force is not None:
                part_force = part_force.clone()
                part_force = part_force - torch.dot(part_force, table_normal) * table_normal
                leg_force_mag = torch.linalg.norm(part_force)
                left_dist = torch.linalg.norm(
                    left_finger_pos - rb_states[part_idxs[self.name]][0][:3]
                )
                right_dist = torch.linalg.norm(
                    right_finger_pos - rb_states[part_idxs[self.name]][0][:3]
                )
                close_to_leg = left_dist < 0.1 and right_dist < 0.1
                left_capsule_dist, right_capsule_dist = self._finger_leg_capsule_distances_env(
                    left_finger_pos, right_finger_pos, rb_states, part_idxs
                )
                capsule_close_to_leg = (
                    left_capsule_dist < 0.07 and right_capsule_dist < 0.07
                )
                close_to_leg = close_to_leg or capsule_close_to_leg
                pinched = leg_force_mag > 1e-3 and close_to_leg and narrow_gripper
            if os.getenv("ANNOTATION_PICK_DEBUG_ROUND_TABLE") == "1":
                print(
                    "[round_table_leg pick_debug] "
                    f"force_mag={self._debug_scalar(leg_force_mag)} "
                    "force_thresh=0.001 "
                    f"force_ok={bool((leg_force_mag is not None and leg_force_mag.item() > 1e-3))} "
                    f"left_center_dist={self._debug_scalar(left_dist)} "
                    f"right_center_dist={self._debug_scalar(right_dist)} "
                    "center_dist_thresh=0.1 "
                    f"left_capsule_dist={self._debug_scalar(left_capsule_dist)} "
                    f"right_capsule_dist={self._debug_scalar(right_capsule_dist)} "
                    "capsule_dist_thresh=0.07 "
                    f"capsule_close_ok={bool(capsule_close_to_leg)} "
                    f"close_ok={bool(close_to_leg)} "
                    f"gripper_width={self._debug_scalar(gripper_width):.6f} "
                    f"gripper_thresh={(config['robot']['max_gripper_width']['round_table'] * 0.8):.6f} "
                    f"gripper_ok={bool(narrow_gripper.item())} "
                    f"pinched={bool(pinched)} "
                    f"pinched_steps={self.skill_pinched_steps}"
                )
            self.skill_pinched_steps = self.skill_pinched_steps + 1 if pinched else 0
            if self.skill_pinched_steps >= 2:
                self.skill_state = "place"
                self.skill_target_ee_pose_robot = self._compute_skill_place_target(
                    ee_pose_robot,
                    rb_states,
                    part_idxs,
                    sim_to_april_mat,
                    april_to_robot,
                    assemble_to,
                )
                self.skill_guidance_point_robot = self.skill_target_ee_pose_robot[:3, 3].clone()
                self.skill_guidance_pose_robot = self.skill_target_ee_pose_robot.clone()
                self.skill_target_gripper_width = self.half_width * 2
        elif self.skill_state == "place":
            self.skill_target_ee_pose_robot = self._compute_skill_place_target(
                ee_pose_robot,
                rb_states,
                part_idxs,
                sim_to_april_mat,
                april_to_robot,
                assemble_to,
            )
            self.skill_guidance_point_robot = self.skill_target_ee_pose_robot[:3, 3].clone()
            self.skill_guidance_pose_robot = self.skill_target_ee_pose_robot.clone()
            self.skill_target_gripper_width = self.half_width * 2
            base_pose_robot = self._part_pose_robot(
                assemble_to, rb_states, part_idxs, sim_to_april_mat, april_to_robot
            )
            leg_pose_robot = self._part_pose_robot(
                self.name, rb_states, part_idxs, sim_to_april_mat, april_to_robot
            )
            xy_error, z_error, ori_error = self._part_place_errors_robot(
                leg_pose_robot,
                base_pose_robot,
                ignore_axis=2,
            )
            xz_ok = xy_error < self.skill_place_part_xz_threshold
            y_ok = z_error < self.skill_place_z_threshold
            ori_ok = ori_error < self.skill_place_part_ori_threshold
            place_ok = xz_ok and y_ok and ori_ok
            if os.getenv("ANNOTATION_PLACE_DEBUG_ROUND_TABLE") == "1":
                blockers = []
                if not xz_ok:
                    blockers.append("xz")
                if not y_ok:
                    blockers.append("y")
                if not ori_ok:
                    blockers.append("ori")
                print(
                    "[round_table_leg place_debug] "
                    f"xz_error={xy_error.item():.6f} "
                    f"xz_thresh={self.skill_place_part_xz_threshold:.6f} "
                    f"xz_ok={bool(xz_ok.item())} "
                    f"y_error={z_error.item():.6f} "
                    f"y_thresh={self.skill_place_z_threshold:.6f} "
                    f"y_ok={bool(y_ok.item())} "
                    f"ori_error={ori_error.item():.6f} "
                    f"ori_thresh={self.skill_place_part_ori_threshold:.6f} "
                    f"ori_ok={bool(ori_ok.item())} "
                    f"place_ok={bool(place_ok.item())} "
                    f"blockers={','.join(blockers) if blockers else 'none'}"
                )
            if place_ok:
                self.skill_state = "insert"
                self.skill_target_ee_pose_robot = self._compute_skill_insert_target(
                    ee_pose_robot,
                    rb_states,
                    part_idxs,
                    sim_to_april_mat,
                    april_to_robot,
                    assemble_to,
                )
                self.skill_guidance_point_robot = self.skill_target_ee_pose_robot[:3, 3].clone()
                self.skill_guidance_pose_robot = self.skill_target_ee_pose_robot.clone()
                self.skill_target_gripper_width = self.half_width * 2
        elif self.skill_state == "insert":
            self.skill_target_ee_pose_robot = self._compute_skill_insert_target(
                ee_pose_robot,
                rb_states,
                part_idxs,
                sim_to_april_mat,
                april_to_robot,
                assemble_to,
            )
            self.skill_guidance_point_robot = self.skill_target_ee_pose_robot[:3, 3].clone()
            self.skill_guidance_pose_robot = self.skill_target_ee_pose_robot.clone()
            self.skill_target_gripper_width = self.half_width * 2
            if (
                gripper_width
                >= config["robot"]["max_gripper_width"]["round_table"] - 0.001
            ):
                self.skill_state = "screw"
                self.skill_target_ee_pose_robot = self._compute_skill_screw_pose_target(
                    rb_states,
                    part_idxs,
                    sim_to_april_mat,
                    april_to_robot,
                    assemble_to,
                )
                self.skill_guidance_pose_robot = self.skill_target_ee_pose_robot.clone()
                self.skill_guidance_point_robot = self._compute_skill_screw_target(
                    rb_states,
                    part_idxs,
                    sim_to_april_mat,
                    april_to_robot,
                    assemble_to,
                )
                self.skill_target_gripper_width = self.reset_gripper_width
        elif self.skill_state == "screw":
            self.skill_target_ee_pose_robot = self._compute_skill_screw_pose_target(
                rb_states,
                part_idxs,
                sim_to_april_mat,
                april_to_robot,
                assemble_to,
            )
            self.skill_guidance_pose_robot = self.skill_target_ee_pose_robot.clone()
            self.skill_guidance_point_robot = self._compute_skill_screw_target(
                rb_states,
                part_idxs,
                sim_to_april_mat,
                april_to_robot,
                assemble_to,
            )
            self.skill_target_gripper_width = self.reset_gripper_width
            if assembled:
                self.skill_state = "done"

        return self.skill_state

    def fsm_step(
        self,
        ee_pos,
        ee_quat,
        gripper_width,
        rb_states,
        part_idxs,
        sim_to_april_mat,
        april_to_robot,
        assemble_to,
    ):
        def rot_mat_tensor(x, y, z, device):
            return torch.tensor(rot_mat([x, y, z], hom=True), device=device).float()

        def rel_rot_mat(s, t):
            s_inv = torch.linalg.inv(s)
            return t @ s_inv

        next_state = self._state

        ee_pose = C.to_homogeneous(ee_pos, C.quat2mat(ee_quat))
        base_pose = C.to_homogeneous(
            rb_states[part_idxs[assemble_to]][0][:3],
            C.quat2mat(rb_states[part_idxs[assemble_to]][0][3:7]),
        )
        leg_pose = C.to_homogeneous(
            rb_states[part_idxs[self.name]][0][:3],
            C.quat2mat(rb_states[part_idxs[self.name]][0][3:7]),
        )

        base_pose = sim_to_april_mat @ base_pose
        leg_pose = sim_to_april_mat @ leg_pose

        margin = rot_mat_tensor(0, -np.pi / 5, 0, ee_pose.device)
        device = ee_pose.device

        def find_leg_pose_x_look_front(leg_pose):
            best_leg_pose = leg_pose.clone()
            tmp_leg_pose = leg_pose
            rot = rot_mat_tensor(0, -np.pi / 2, 0, device)
            for i in range(3):
                tmp_leg_pose = tmp_leg_pose @ rot
                if best_leg_pose[0, 0] < tmp_leg_pose[0, 0]:
                    best_leg_pose = tmp_leg_pose
            return best_leg_pose

        if self._state == "reach_leg_floor_xy":
            leg_pose = self._find_down_z(leg_pose).clone().to(device)
            # Margin for leg pose
            leg_pose = (
                torch.tensor(get_mat([0, 0.043, 0], [0, 0, 0]), device=device)
                @ leg_pose
            )
            rot = rot_mat_tensor(np.pi / 2, -np.pi / 2, 0, device)
            pos = leg_pose[:4, 3]
            target_pos = (april_to_robot @ pos)[:3]
            target_ori = ee_pose[:3, :3]
            target_pos[2] = ee_pos[2]
            target = self.add_noise_first_target(
                C.to_homogeneous(target_pos, target_ori),
                ori_noise=torch.tensor([0, 0, 0, 1], device=device),
            )
            if self.satisfy(ee_pose, target, pos_error_threshold=0.02):
                self.prev_pose = target.clone()
                self.prev_pose[2, 3] -= 0.02
                next_state = "reach_leg_ori"
        elif self._state == "reach_leg_ori":
            rot = rot_mat_tensor(np.pi / 2, -np.pi / 2, 0, device)
            theta_y = torch.acos(leg_pose[1, 1]).detach().cpu().numpy()
            sign = 1 if leg_pose[0, 1] > 0 else -1
            target_ori = (
                rot_mat_tensor(0, 0, sign * theta_y, device)
                @ margin
                @ april_to_robot
                @ rot
            )[:3, :3]
            # Get the y-axis rotation.
            target_pos = (
                april_to_robot
                @ torch.tensor(get_mat([0, 0.043, 0], [0, 0, 0]), device=device)
                @ leg_pose[:4, 3]
            )[:3]
            target_pos[2] = ee_pos[2]
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(ee_pose, target, pos_error_threshold=0.015):
                self.prev_pose = target
                next_state = "reach_leg_floor_z"
        elif self._state == "reach_leg_floor_z":
            rot = rot_mat_tensor(np.pi / 2, -np.pi / 2, 0, device)
            theta_y = torch.acos(leg_pose[1, 1]).detach().cpu().numpy()
            sign = 1 if leg_pose[0, 1] > 0 else -1
            target_ori = (
                rot_mat_tensor(0, 0, sign * theta_y, device)
                @ margin
                @ april_to_robot
                @ rot
            )[:3, :3]
            # Get the y-axis rotation.
            target_pos = (
                april_to_robot
                @ torch.tensor(get_mat([0, 0.043, 0], [0, 0, 0]), device=device)
                @ leg_pose[:4, 3]
            )[:3]
            target_pos[2] += 0.01  # Margin.
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(ee_pose, target, pos_error_threshold=0.007):
                self.prev_pose = target
                next_state = "pick_leg"
        elif self._state == "pick_leg":
            target = self.prev_pose
            self.gripper_action = 1
            if self.gripper_less(gripper_width, 2 * self.half_width + 0.001):
                self.prev_pose = target
                next_state = "lift_up"
        elif self._state == "lift_up":
            target_pos = self.prev_pose[:3, 3] + torch.tensor(
                [0, 0, 0.13], device=device
            )
            target_ori = ee_pose[:3, :3]
            target = self.add_noise_first_target(
                C.to_homogeneous(target_pos, target_ori)
            )
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.02, ori_error_threshold=0.3
            ):
                self.prev_pose = target
                next_state = "move_center"
        elif self._state == "move_center":
            target_pos = torch.tensor([0.5, 0.10, 0.13], device=device)
            target_ori = self.prev_pose[:3, :3]
            target = self.add_noise_first_target(
                C.to_homogeneous(target_pos, target_ori)
            )
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.02, ori_error_threshold=0.3
            ):
                self.prev_pose = target
                next_state = "match_leg_ori"
        elif self._state == "match_leg_ori":
            # target_ori = (margin @ rot_mat_tensor(np.pi, 0, 0, device))[:3, :3]
            target_ori = (rot_mat_tensor(np.pi, 0, 0, device))[:3, :3]
            target_pos = torch.tensor([0.57, 0.10, 0.13], device=device)
            target = self.add_noise_first_target(
                C.to_homogeneous(target_pos, target_ori)
            )
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.02, ori_error_threshold=0.3
            ):
                self.prev_pose = target
                next_state = "reach_base_xy"
        elif self._state == "reach_base_xy":
            leg_pose_robot = april_to_robot @ leg_pose
            leg_pose_robot = find_leg_pose_x_look_front(leg_pose_robot)
            base_hole_pose_robot = (
                april_to_robot
                @ base_pose
                @ torch.tensor(
                    get_mat(self.default_assembled_pose[:3, 3], [0.0, 0.0, 0.0]),
                    device=device,
                )
            )
            target_hole_pose_robot = torch.tensor(
                [
                    [1.0, 0.0, 0.0, base_hole_pose_robot[0, 3]],
                    [0.0, 0.0, -1.0, base_hole_pose_robot[1, 3]],
                    [0.0, 1.0, 0.0, self.prev_pose[2, 3]],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                device=device,
            )
            rel = rel_rot_mat(leg_pose_robot, target_hole_pose_robot)
            target = rel @ ee_pose
            if self.satisfy(
                ee_pose,
                target,
                pos_error_threshold=0.0,
                ori_error_threshold=0.3,
                max_len=20,
            ):
                self.prev_pose = target
                next_state = "reach_base_z"
        elif self._state == "reach_base_z":
            leg_pose_robot = april_to_robot @ leg_pose
            leg_pose_robot = find_leg_pose_x_look_front(leg_pose_robot)
            base_hole_pose_robot = (
                april_to_robot
                @ base_pose
                @ torch.tensor(
                    get_mat(self.default_assembled_pose[:3, 3], [0.0, 0.0, 0.0]),
                    device=device,
                )
            )
            target_ori = torch.tensor(
                get_mat([0, 0, 0], [np.pi / 2, 0, -np.pi / 4]), device=device
            )[:3, :3]
            target_pos = base_hole_pose_robot[:3, 3]
            target_hole_pose_robot = C.to_homogeneous(target_pos, target_ori)

            rel = rel_rot_mat(leg_pose_robot, target_hole_pose_robot)
            target = rel @ ee_pose
            target[2] += 0.03  # Margin.
            if self.satisfy(
                ee_pose,
                target,
                pos_error_threshold=0.000,
                ori_error_threshold=0.0,
                max_len=30,
            ):
                self.prev_pose = target
                next_state = "insert_wait"
        elif self._state == "insert_wait":
            target = self.prev_pose
            self.gripper_action = -1
            if self.gripper_greater(
                gripper_width,
                config["robot"]["max_gripper_width"]["round_table"] - 0.001,
            ):
                next_state = "insert"
        elif self._state == "insert":
            # Dummy transition state for skill complete.
            target = self.prev_pose
            next_state = "pre_grasp"
        elif self._state == "release":
            target = self.prev_pose
            self.gripper_action = -1
            if self.gripper_greater(
                gripper_width,
                config["robot"]["max_gripper_width"]["round_table"] - 0.001,
            ):
                next_state = "pre_grasp"
        elif self._state == "pre_grasp":
            target_ori = rot_mat_tensor(np.pi, 0, 0, device)[:3, :3]
            target_pos = (april_to_robot @ leg_pose[:4, 3])[:3]
            target_pos[2] += self.grasp_margin_z
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(ee_pose, target, max_len=30):
                self.prev_pose = target
                next_state = "screw_grasp"
        elif self._state == "screw_grasp":
            target = self.prev_pose
            self.gripper_action = 1
            if self.gripper_less(gripper_width, 2 * self.half_width + 0.001):
                self.prev_pose = target
                next_state = "screw"
        elif self._state == "screw":
            target_ori = rot_mat_tensor(np.pi, 0, -np.pi / 2 - np.pi / 36, device)[
                :3, :3
            ]
            target_pos = (ee_pos)[:3]
            target_pos[2] -= 0.005
            target = C.to_homogeneous(target_pos, target_ori)

            if self.satisfy(ee_pose, target, ori_error_threshold=0.3):
                self.prev_pose = target
                next_state = "release"

        skill_complete = self.may_transit_state(next_state)

        return (
            target[:3, 3],
            C.mat2quat(target[:3, :3]),
            torch.tensor([self.gripper_action], device=device),
            skill_complete,
        )

    def state_no_noise(self):
        return self._state in [
            "insert_wait",
            "insert_release",
            "insert",
        ]
