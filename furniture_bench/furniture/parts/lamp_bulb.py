import numpy as np
import torch
import numpy.typing as npt

from furniture_bench.furniture.parts.leg import Leg
from furniture_bench.utils.pose import get_mat, rot_mat
from furniture_bench.config import config
import furniture_bench.controllers.control_utils as C
import furniture_bench.utils.transform as T


class LampBulb(Leg):
    _gripper_width_key = "lamp"

    def __init__(self, part_config, part_idx):
        self.half_width = 0.0175
        self.tag_offset = 0.0175
        self.reset_x_len = 0.057
        self.reset_y_len = 0.13

        super().__init__(part_config, part_idx)

        self.reset_gripper_width = 0.07
        self.grasp_margin_x = 0.043
        self.grasp_margin_z = 0.053

    def reset(self):
        self.prev_pose = None
        self._state = "reach_bulb_floor_xy"
        self.gripper_action = -1
        self.reset_skill_state()

    def reset_skill_state(self):
        super().reset_skill_state()
        self.skill_place_xy_threshold = 0.008
        self.skill_place_z_threshold = 0.05
        self.skill_place_ori_threshold = 0.5

    def _find_down_z(self, mat):
        max_mat = mat.clone()
        for _ in range(4):
            if max_mat[2, 2] < mat[2, 2]:
                max_mat = mat.clone()
            mat = mat @ torch.tensor(
                T.rotmat2hom(rot_mat([0, np.pi / 2, 0]))
            ).float().to(mat.device)
        return max_mat

    def _find_bulb_pose_x_look_front_skill(self, bulb_pose, device):
        best_bulb_pose = bulb_pose.clone()
        tmp_bulb_pose = bulb_pose
        rot = torch.tensor(rot_mat([0, -np.pi / 2, 0], hom=True), device=device).float()
        for _ in range(3):
            tmp_bulb_pose = tmp_bulb_pose @ rot
            if best_bulb_pose[0, 0] < tmp_bulb_pose[0, 0]:
                best_bulb_pose = tmp_bulb_pose
        return best_bulb_pose

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
        bulb_pose_env = C.to_homogeneous(
            rb_states[part_idxs[self.name]][0][:3],
            C.quat2mat(rb_states[part_idxs[self.name]][0][3:7]),
        )
        base_pose_april = sim_to_april_mat @ base_pose_env
        bulb_pose_april = sim_to_april_mat @ bulb_pose_env
        base_pose_robot = april_to_robot @ base_pose_april
        bulb_pose_robot = april_to_robot @ bulb_pose_april
        base_hole_pose_robot = (
            base_pose_robot
            @ torch.tensor(
                get_mat(self.default_assembled_pose[:3, 3], [0.0, 0.0, 0.0]),
                device=device,
            )
        )
        target_hole_pose_robot = torch.tensor(
            [
                [1.0, 0.0, 0.0, base_hole_pose_robot[0, 3]],
                [0.0, 0.0, -1.0, base_hole_pose_robot[1, 3]],
                [0.0, 1.0, 0.0, base_pose_robot[2, 3] + 0.09],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
        )
        rel_robot = target_hole_pose_robot @ torch.linalg.inv(bulb_pose_robot)
        target_ee_pose_robot = rel_robot @ ee_pose_robot
        self.skill_target_part_pose_robot = target_hole_pose_robot.clone()
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
        bulb_pose_env = C.to_homogeneous(
            rb_states[part_idxs[self.name]][0][:3],
            C.quat2mat(rb_states[part_idxs[self.name]][0][3:7]),
        )
        bulb_pose_april = sim_to_april_mat @ bulb_pose_env
        bulb_pose_robot = april_to_robot @ bulb_pose_april
        long_axis_idx = 1 if self.reset_y_len >= self.reset_x_len else 0
        long_axis_len = self.reset_y_len if long_axis_idx == 1 else self.reset_x_len
        long_axis_robot = bulb_pose_robot[:3, long_axis_idx]
        long_axis_offset_robot = long_axis_robot * (long_axis_len * 0.25)
        return (bulb_pose_robot[:3, 3] + long_axis_offset_robot).clone()

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

        # Reverse edges place/insert/screw -> pick: if the bulb was released
        # during assembly (dropped) and is no longer seated at the base,
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
            pinched = False
            left_dist = None
            right_dist = None
            close_to_bulb = False
            narrow_gripper = gripper_width < config["robot"]["max_gripper_width"]["lamp"] * 0.9
            if part_force is not None:
                part_force = part_force.clone()
                part_force = part_force - torch.dot(part_force, table_normal) * table_normal
                left_dist = torch.linalg.norm(
                    left_finger_pos - rb_states[part_idxs[self.name]][0][:3]
                )
                right_dist = torch.linalg.norm(
                    right_finger_pos - rb_states[part_idxs[self.name]][0][:3]
                )
                close_to_bulb = left_dist < 0.12 and right_dist < 0.12
                pinched = torch.linalg.norm(part_force) > 1e-3 and close_to_bulb and narrow_gripper
            # print(
            #     "[lamp_bulb pick_debug] "
            #     f"force_mag={(torch.linalg.norm(part_force).item() if part_force is not None else None)} "
            #     "force_thresh=0.001 "
            #     f"force_ok={(part_force is not None and torch.linalg.norm(part_force).item() > 1e-3)} "
            #     f"left_dist={(left_dist.item() if left_dist is not None else None)} "
            #     f"right_dist={(right_dist.item() if right_dist is not None else None)} "
            #     "dist_thresh=0.12 "
            #     f"close_ok={close_to_bulb} "
            #     f"gripper_width={gripper_width.item():.6f} "
            #     f"gripper_thresh={(config['robot']['max_gripper_width']['lamp'] * 0.9):.6f} "
            #     f"gripper_ok={bool(narrow_gripper.item())} "
            #     f"pinched={pinched} "
            #     f"pinched_steps={self.skill_pinched_steps}"
            # )
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
            base_pose_robot = self._part_pose_robot(
                assemble_to, rb_states, part_idxs, sim_to_april_mat, april_to_robot
            )
            bulb_pose_robot = self._part_pose_robot(
                self.name, rb_states, part_idxs, sim_to_april_mat, april_to_robot
            )
            xy_error, z_error, ori_error = self._part_place_errors_robot(
                bulb_pose_robot,
                base_pose_robot,
                ignore_axis=1,
            )
            if (
                xy_error < self.skill_place_part_xz_threshold
                and z_error < self.skill_place_z_threshold
                and ori_error < self.skill_place_part_ori_threshold
            ):
                self.skill_state = "insert"
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
            if gripper_width >= config["robot"]["max_gripper_width"]["lamp"] - 0.001:
                self.skill_state = "screw"
                self.skill_guidance_point_robot = self._compute_skill_screw_target(
                    rb_states,
                    part_idxs,
                    sim_to_april_mat,
                    april_to_robot,
                    assemble_to,
                )
        elif self.skill_state == "screw":
            self.skill_guidance_point_robot = self._compute_skill_screw_target(
                rb_states,
                part_idxs,
                sim_to_april_mat,
                april_to_robot,
                assemble_to,
            )
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
        bulb_pose = C.to_homogeneous(
            rb_states[part_idxs[self.name]][0][:3],
            C.quat2mat(rb_states[part_idxs[self.name]][0][3:7]),
        )

        base_pose = sim_to_april_mat @ base_pose
        bulb_pose = sim_to_april_mat @ bulb_pose

        margin = rot_mat_tensor(0, -np.pi / 5, 0, ee_pose.device)
        device = ee_pose.device

        def find_bulb_pose_x_look_front(bulb_pose):
            best_bulb_pose = bulb_pose.clone()
            tmp_bulb_pose = bulb_pose
            rot = rot_mat_tensor(0, -np.pi / 2, 0, device)
            for i in range(3):
                tmp_bulb_pose = tmp_bulb_pose @ rot
                if best_bulb_pose[0, 0] < tmp_bulb_pose[0, 0]:
                    best_bulb_pose = tmp_bulb_pose
            return best_bulb_pose

        if self._state == "reach_bulb_floor_xy":
            bulb_pose = self._find_down_z(bulb_pose).clone().to(device)
            bulb_pose = (
                torch.tensor(get_mat([0, 0.043, 0], [0, 0, 0]), device=device)
                @ bulb_pose
            )
            rot = rot_mat_tensor(np.pi / 2, -np.pi / 2, 0, device)
            pos = bulb_pose[:4, 3]
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
                next_state = "reach_bulb_ori"
        elif self._state == "reach_bulb_ori":
            rot = rot_mat_tensor(np.pi / 2, -np.pi / 2, 0, device)
            theta_y = torch.acos(bulb_pose[1, 1]).detach().cpu().numpy()
            sign = 1 if bulb_pose[0, 1] > 0 else -1
            target_ori = (
                rot_mat_tensor(0, 0, sign * theta_y, device)
                @ margin
                @ april_to_robot
                @ rot
            )[:3, :3]
            target_pos = (
                april_to_robot
                @ torch.tensor(get_mat([0, 0.043, 0], [0, 0, 0]), device=device)
                @ bulb_pose[:4, 3]
            )[:3]
            target_pos[2] = ee_pos[2]
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(ee_pose, target, pos_error_threshold=0.015):
                self.prev_pose = target
                next_state = "reach_bulb_floor_z"
        elif self._state == "reach_bulb_floor_z":
            rot = rot_mat_tensor(np.pi / 2, -np.pi / 2, 0, device)
            theta_y = torch.acos(bulb_pose[1, 1]).detach().cpu().numpy()
            sign = 1 if bulb_pose[0, 1] > 0 else -1
            target_ori = (
                rot_mat_tensor(0, 0, sign * theta_y, device)
                @ margin
                @ april_to_robot
                @ rot
            )[:3, :3]
            target_pos = (
                april_to_robot
                @ torch.tensor(get_mat([0, 0.043, 0], [0, 0, 0]), device=device)
                @ bulb_pose[:4, 3]
            )[:3]
            target_pos[2] += 0.01
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
                [0, 0, 0.17], device=device
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
            target_pos = torch.tensor([0.5, 0.10, 0.17], device=device)
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
            target_ori = (rot_mat_tensor(np.pi, 0, 0, device))[:3, :3]
            target_pos = torch.tensor([0.57, 0.10, 0.17], device=device)
            target = self.add_noise_first_target(
                C.to_homogeneous(target_pos, target_ori)
            )
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.02, ori_error_threshold=0.3
            ):
                self.prev_pose = target
                next_state = "reach_base_xy"
        elif self._state == "reach_base_xy":
            bulb_pose_robot = april_to_robot @ bulb_pose
            bulb_pose_robot = find_bulb_pose_x_look_front(bulb_pose_robot)
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
            rel = rel_rot_mat(bulb_pose_robot, target_hole_pose_robot)
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
            bulb_pose_robot = april_to_robot @ bulb_pose
            bulb_pose_robot = find_bulb_pose_x_look_front(bulb_pose_robot)
            base_hole_pose_robot = (
                april_to_robot
                @ base_pose
                @ torch.tensor(
                    get_mat(self.default_assembled_pose[:3, 3], [0.0, 0.0, 0.0]),
                    device=device,
                )
            )
            target_ori = torch.tensor(
                get_mat([0, 0, 0], [np.pi / 2, 0, np.pi / 4]), device=device
            )[:3, :3]
            target_pos = base_hole_pose_robot[:3, 3]
            target_hole_pose_robot = C.to_homogeneous(target_pos, target_ori)

            rel = rel_rot_mat(bulb_pose_robot, target_hole_pose_robot)
            target = rel @ ee_pose
            target[2] += 0.03
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.000, ori_error_threshold=0.0, max_len=30
            ):
                self.prev_pose = target
                next_state = "insert_wait"
        elif self._state == "insert_wait":
            target = self.prev_pose
            self.gripper_action = -1
            if self.gripper_greater(
                gripper_width,
                config["robot"]["max_gripper_width"]["square_table"] - 0.001,
            ):
                next_state = "insert"
        elif self._state == "insert":
            target = self.prev_pose
            next_state = "pre_grasp"
        elif self._state == "release":
            target = self.prev_pose
            self.gripper_action = -1
            if self.gripper_greater(
                gripper_width,
                config["robot"]["max_gripper_width"]["square_table"] - 0.001,
            ):
                next_state = "pre_grasp"
        elif self._state == "pre_grasp":
            target_ori = rot_mat_tensor(np.pi, 0, 0, device)[:3, :3]
            target_pos = (april_to_robot @ bulb_pose[:4, 3])[:3]
            target_pos[2] += self.grasp_margin_z
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(ee_pose, target):
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
