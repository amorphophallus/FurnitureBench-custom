import numpy as np
import torch

from furniture_bench.utils.pose import get_mat, rot_mat
from furniture_bench.furniture.parts.part import Part
import furniture_bench.utils.transform as T
import furniture_bench.controllers.control_utils as C
from furniture_bench.config import config


class RoundTableBase(Part):
    def __init__(self, part_config, part_idx):
        super().__init__(part_config, part_idx)
        self.reset_x_len = 0.02
        self.reset_y_len = 0.02

        self.rel_pose_from_center[self.tag_ids[0]] = get_mat(
            [0, -0.045625, 0], [0, 0, 0]
        )
        self.rel_pose_from_center[self.tag_ids[1]] = get_mat(
            [-0.0456255, 00, 0], [0, 0, -np.pi / 2]
        )
        self.rel_pose_from_center[self.tag_ids[2]] = get_mat(
            [0, 0.045625, 0], [0, 0, np.pi]
        )
        self.rel_pose_from_center[self.tag_ids[3]] = get_mat(
            [0.045625, 0, 0], [0, 0, np.pi / 2]
        )
        self.rel_pose_from_center[self.tag_ids[4]] = get_mat(
            [0, -0.045625, 0.01], [0, np.pi, 0]
        )
        self.rel_pose_from_center[self.tag_ids[5]] = get_mat(
            [-0.045625, 0, 0.01], [0, np.pi, -np.pi / 2]
        )
        self.rel_pose_from_center[self.tag_ids[6]] = get_mat(
            [0, 0.045625, 0.01], [0, np.pi, np.pi]
        )
        self.rel_pose_from_center[self.tag_ids[7]] = get_mat(
            [0.045625, 0, 0.01], [0, np.pi, np.pi / 2]
        )
        self._state = "move_center"
        self.skill_complete_next_states = ["lift_up_base", "insert_base"]
        self.half_width = 0.02
        self.grasp_margin_z = 0.006
        self.init_ee_pos = None

        self.base_grip_width = 0.04

    def is_in_reset_ori(self, pose, from_skill, ori_bound):
        return pose[2, 2] < -ori_bound

    def heading_down(self, pose):
        return pose[2, 2] > self.reset_ori_bound

    def reset(self):
        self.pre_assemble_done = False
        self._state = "move_up"
        self.gripper_action = -1
        self.reset_skill_state()

    def reset_skill_state(self):
        self.skill_state = "pick"
        self.skill_target = None
        self.skill_guidance_point = None
        self.skill_pinched_steps = 0
        self.pick_force_threshold = 1e-3
        self.pick_distance_threshold = 0.1
        self.pick_gripper_ratio_threshold = 0.8
        self.skill_place_xy_threshold = 0.008
        self.skill_place_z_threshold = 0.05
        self.skill_place_ori_threshold = 0.5

    def get_skill_label(self):
        if self.skill_state == "done":
            return None
        return self.skill_state

    def get_guidance_point(self):
        return self.skill_guidance_point

    def _ori_error_no_yaw(self, current_rot, target_rot):
        def remove_yaw(rot):
            yaw = torch.atan2(rot[1, 0], rot[0, 0])
            cy = torch.cos(-yaw)
            sy = torch.sin(-yaw)
            rot_z = torch.stack(
                [
                    torch.stack([cy, -sy, torch.tensor(0.0, device=rot.device)]),
                    torch.stack([sy, cy, torch.tensor(0.0, device=rot.device)]),
                    torch.tensor([0.0, 0.0, 1.0], device=rot.device),
                ]
            )
            return rot_z @ rot

        current_ny = remove_yaw(current_rot)
        target_ny = remove_yaw(target_rot)
        return (current_ny - target_ny).abs().sum()

    def _compute_skill_pick_target(
        self,
        ee_pose,
        rb_states,
        part_idxs,
        sim_to_april_mat,
        april_to_robot,
    ):
        base_pose = C.to_homogeneous(
            rb_states[part_idxs[self.name]][0][:3],
            C.quat2mat(rb_states[part_idxs[self.name]][0][3:7]),
        )
        base_pose = sim_to_april_mat @ base_pose
        theta_y = torch.acos(base_pose[1, 1]).detach().cpu().numpy()
        if base_pose[0, 1] < 0:
            theta = np.pi - theta_y + np.pi / 4
        else:
            theta = theta_y - 3 / 4 * np.pi
        rot = torch.tensor(rot_mat([np.pi, 0, 0], hom=True), device=ee_pose.device).float()
        target_ori = (
            torch.tensor(rot_mat([0, 0, theta], hom=True), device=ee_pose.device).float()
            @ rot
        )[:3, :3]
        pos = base_pose[:4, 3]
        target_pos = (april_to_robot @ pos)[:3]
        target_pos[2] = ee_pose[2, 3]
        return C.to_homogeneous(target_pos, target_ori)

    def _compute_skill_pick_guidance_point(
        self,
        rb_states,
        part_idxs,
        sim_to_april_mat,
        april_to_robot,
    ):
        base_pose = C.to_homogeneous(
            rb_states[part_idxs[self.name]][0][:3],
            C.quat2mat(rb_states[part_idxs[self.name]][0][3:7]),
        )
        base_pose = sim_to_april_mat @ base_pose
        target_pos = (april_to_robot @ base_pose[:4, 3])[:3]
        target_pos[2] = (april_to_robot @ base_pose)[2, 3]
        return target_pos

    def _compute_skill_place_target(
        self,
        ee_pose,
        rb_states,
        part_idxs,
        sim_to_april_mat,
        april_to_robot,
        assemble_to,
    ):
        device = ee_pose.device
        leg_pose = C.to_homogeneous(
            rb_states[part_idxs[assemble_to]][0][:3],
            C.quat2mat(rb_states[part_idxs[assemble_to]][0][3:7]),
        )
        leg_pose = sim_to_april_mat @ leg_pose
        base_screw_pose_robot = (
            april_to_robot
            @ leg_pose
            @ torch.tensor(
                get_mat(self.default_assembled_pose[:3, 3], [0.0, 0.0, 0.0]),
                device=device,
            )
        )
        target = ee_pose.clone()
        target[:3, 3] = base_screw_pose_robot[:3, 3]
        return target

    def _compute_skill_insert_target(
        self,
        ee_pose,
        rb_states,
        part_idxs,
        sim_to_april_mat,
        april_to_robot,
        assemble_to,
    ):
        device = ee_pose.device
        leg_pose = C.to_homogeneous(
            rb_states[part_idxs[assemble_to]][0][:3],
            C.quat2mat(rb_states[part_idxs[assemble_to]][0][3:7]),
        )
        leg_pose = sim_to_april_mat @ leg_pose
        base_screw_pose_robot = (
            april_to_robot
            @ leg_pose
            @ torch.tensor(
                get_mat(self.default_assembled_pose[:3, 3], [0.0, 0.0, 0.0]),
                device=device,
            )
        )
        target = ee_pose.clone()
        target[:3, 3] = base_screw_pose_robot[:3, 3]
        target[2, 3] += 0.02
        return target

    def _compute_skill_screw_target(
        self,
        rb_states,
        part_idxs,
        sim_to_april_mat,
        april_to_robot,
        assemble_to,
    ):
        leg_pose = C.to_homogeneous(
            rb_states[part_idxs[assemble_to]][0][:3],
            C.quat2mat(rb_states[part_idxs[assemble_to]][0][3:7]),
        )
        leg_pose = sim_to_april_mat @ leg_pose
        base_screw_pose_robot = (
            april_to_robot
            @ leg_pose
            @ torch.tensor(
                get_mat(self.default_assembled_pose[:3, 3], [0.0, 0.0, 0.0]),
                device=leg_pose.device,
            )
        )
        return base_screw_pose_robot[:3, 3].clone()

    def _compute_skill_object_target(
        self,
        rb_states,
        part_idxs,
        sim_to_april_mat,
        april_to_robot,
    ):
        base_pose = C.to_homogeneous(
            rb_states[part_idxs[self.name]][0][:3],
            C.quat2mat(rb_states[part_idxs[self.name]][0][3:7]),
        )
        base_pose = sim_to_april_mat @ base_pose
        return (april_to_robot @ base_pose)[:3, 3].clone()

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
        ee_pose = C.to_homogeneous(ee_pos, C.quat2mat(ee_quat))
        table_normal = torch.tensor([0.0, 0.0, 1.0], device=ee_pos.device)
        grasp_axis = right_finger_pos - left_finger_pos
        grasp_axis[2] = 0.0
        if torch.linalg.norm(grasp_axis) < 1e-6:
            grasp_axis = torch.tensor([1.0, 0.0, 0.0], device=ee_pos.device)

        if self.skill_state == "pick":
            self.skill_guidance_point = self._compute_skill_pick_guidance_point(
                rb_states,
                part_idxs,
                sim_to_april_mat,
                april_to_robot,
            )
            pinched = False
            base_force_mag = None
            left_dist = None
            right_dist = None
            close_to_base = False
            narrow_gripper = (
                gripper_width
                < config["robot"]["max_gripper_width"]["round_table"]
                * self.pick_gripper_ratio_threshold
            )
            if part_force is not None:
                part_force = part_force.clone()
                part_force = part_force - torch.dot(part_force, table_normal) * table_normal
                base_force_mag = torch.linalg.norm(part_force)
                left_dist = torch.linalg.norm(
                    left_finger_pos - rb_states[part_idxs[self.name]][0][:3]
                )
                right_dist = torch.linalg.norm(
                    right_finger_pos - rb_states[part_idxs[self.name]][0][:3]
                )
                close_to_base = (
                    left_dist < self.pick_distance_threshold
                    and right_dist < self.pick_distance_threshold
                )
                pinched = (
                    base_force_mag > self.pick_force_threshold
                    and close_to_base
                    and narrow_gripper
                )
            print(
                "[round_table_base pick_debug] "
                f"force_mag={(base_force_mag.item() if base_force_mag is not None else None)} "
                f"force_thresh={self.pick_force_threshold:.6f} "
                f"force_ok={(base_force_mag is not None and base_force_mag.item() > self.pick_force_threshold)} "
                f"left_dist={(left_dist.item() if left_dist is not None else None)} "
                f"right_dist={(right_dist.item() if right_dist is not None else None)} "
                f"dist_thresh={self.pick_distance_threshold:.6f} "
                f"close_ok={close_to_base} "
                f"gripper_width={gripper_width.item():.6f} "
                f"gripper_thresh={(config['robot']['max_gripper_width']['round_table'] * self.pick_gripper_ratio_threshold):.6f} "
                f"gripper_ok={bool(narrow_gripper.item())} "
                f"pinched={pinched} "
                f"pinched_steps={self.skill_pinched_steps}"
            )
            self.skill_pinched_steps = self.skill_pinched_steps + 1 if pinched else 0
            if self.skill_pinched_steps >= 2:
                self.skill_state = "place"
                self.skill_target = self._compute_skill_place_target(
                    ee_pose,
                    rb_states,
                    part_idxs,
                    sim_to_april_mat,
                    april_to_robot,
                    assemble_to,
                )
                self.skill_guidance_point = self.skill_target[:3, 3].clone()
        elif self.skill_state == "place":
            self.skill_target = self._compute_skill_place_target(
                ee_pose,
                rb_states,
                part_idxs,
                sim_to_april_mat,
                april_to_robot,
                assemble_to,
            )
            self.skill_guidance_point = self.skill_target[:3, 3].clone()
            xy_error = (ee_pose[:2, 3] - self.skill_target[:2, 3]).abs().sum()
            z_error = (ee_pose[2, 3] - self.skill_target[2, 3]).abs()
            ori_error = self._ori_error_no_yaw(
                ee_pose[:3, :3], self.skill_target[:3, :3]
            )
            place_ok = (
                xy_error < self.skill_place_xy_threshold
                and z_error < self.skill_place_z_threshold
                and ori_error < self.skill_place_ori_threshold
            )
            print(
                "[round_table_base place_debug] "
                f"xy_error={xy_error.item():.6f} "
                f"xy_thresh={self.skill_place_xy_threshold:.6f} "
                f"z_error={z_error.item():.6f} "
                f"z_thresh={self.skill_place_z_threshold:.6f} "
                f"ori_error={ori_error.item():.6f} "
                f"ori_thresh={self.skill_place_ori_threshold:.6f} "
                f"place_ok={place_ok}"
            )
            if (
                xy_error < self.skill_place_xy_threshold
                and z_error < self.skill_place_z_threshold
                and ori_error < self.skill_place_ori_threshold
            ):
                self.skill_state = "insert"
        elif self.skill_state == "insert":
            self.skill_target = self._compute_skill_insert_target(
                ee_pose,
                rb_states,
                part_idxs,
                sim_to_april_mat,
                april_to_robot,
                assemble_to,
            )
            self.skill_guidance_point = self.skill_target[:3, 3].clone()
            if (
                gripper_width
                >= config["robot"]["max_gripper_width"]["round_table"] - 0.001
            ):
                self.skill_state = "screw"
                self.skill_guidance_point = self.skill_target[:3, 3].clone()
        elif self.skill_state == "screw":
            if self.skill_target is not None:
                self.skill_guidance_point = self.skill_target[:3, 3].clone()
            if assembled:
                self.skill_state = "done"

        return self.skill_state

    def pre_assemble(
        self,
        ee_pos,
        ee_quat,
        gripper_width,
        rb_states,
        part_idxs,
        sim_to_april_mat,
        april_to_robot,
    ):
        next_state = self._state

        ee_pose = C.to_homogeneous(ee_pos, C.quat2mat(ee_quat))
        base_pose = C.to_homogeneous(
            rb_states[part_idxs["round_table_base"]][0][:3],
            C.quat2mat(rb_states[part_idxs["round_table_base"]][0][3:7]),
        )

        base_pose = sim_to_april_mat @ base_pose
        device = base_pose.device

        if self.init_ee_pos is None:
            self.init_ee_pos = ee_pos.clone()

        if self._state == "move_up":
            self.gripper_action = -1
            target_pos = self.init_ee_pos + torch.tensor([0, 0, 0.03], device=device)
            target_ori = ee_pose[:3, :3]
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(
                ee_pose,
                target,
                pos_error_threshold=0.00,
                ori_error_threshold=0.0,
                max_len=20,
            ):
                self.prev_pose = target
                next_state = "move_center"
        if self._state == "move_center":
            z = ee_pose[2, 3]
            target_pos = torch.tensor([self.prev_pose[0, 3], -0.05, z], device=device)
            target_ori = self.prev_pose[:3, :3]
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.02, ori_error_threshold=0.3
            ):
                self.prev_pose = target
                next_state = "reach_base_floor_xy"
                self.pre_assemble_done = True
                target = self.prev_pose

        skill_complete = self.may_transit_state(next_state)

        return (
            target[:3, 3],
            C.mat2quat(target[:3, :3]),
            torch.tensor([self.gripper_action]).to(ee_pos.device),
            skill_complete,
        )

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
        next_state = self._state

        ee_pose = C.to_homogeneous(ee_pos, C.quat2mat(ee_quat))
        leg_pose = C.to_homogeneous(
            rb_states[part_idxs[assemble_to]][0][:3],
            C.quat2mat(rb_states[part_idxs[assemble_to]][0][3:7]),
        )
        base_pose = C.to_homogeneous(
            rb_states[part_idxs[self.name]][0][:3],
            C.quat2mat(rb_states[part_idxs[self.name]][0][3:7]),
        )

        leg_pose = sim_to_april_mat @ leg_pose
        base_pose = sim_to_april_mat @ base_pose

        device = ee_pose.device

        if self._state == "reach_base_floor_xy":
            rot = torch.tensor(rot_mat([np.pi, 0, 0], hom=True), device=device).float()
            theta_y = torch.acos(base_pose[1, 1]).detach().cpu().numpy()
            if base_pose[0, 1] < 0:
                theta = np.pi - theta_y + np.pi / 4
            else:
                theta = theta_y - 3 / 4 * np.pi
            target_ori = (
                torch.tensor(rot_mat([0, 0, theta], hom=True), device=device).float()
                @ rot
            )[:3, :3]
            pos = base_pose[:4, 3]
            target_pos = (april_to_robot @ pos)[:3]
            target_pos[2] = ee_pos[2]  # Keep the z.
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(
                ee_pose,
                target,
                pos_error_threshold=0.0,
                ori_error_threshold=0,
                max_len=30,
            ):
                self.prev_pose = target.clone()
                next_state = "reach_base_floor_z"
        elif self._state == "reach_base_floor_z":
            rot = torch.tensor(rot_mat([np.pi, 0, 0], hom=True), device=device).float()
            theta_y = torch.acos(base_pose[1, 1]).detach().cpu().numpy()
            if base_pose[0, 1] < 0:
                theta = np.pi - theta_y + np.pi / 4
            else:
                theta = theta_y - 3 / 4 * np.pi
            target_ori = (
                torch.tensor(rot_mat([0, 0, theta], hom=True), device=device).float()
                @ rot
            )[:3, :3]

            pos = base_pose[:4, 3]
            target_pos = (april_to_robot @ pos)[:3]
            target_pos[2] += 0.010  # margin.
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.00, ori_error_threshold=0.0
            ):
                self.prev_pose = target
                next_state = "pick_base"
        elif self._state == "pick_base":
            target = self.prev_pose
            self.gripper_action = 1
            if self.gripper_less(
                gripper_width, self.base_grip_width + 0.001, cnt_max=20
            ):
                self.prev_pose = target
                next_state = "lift_up_base"
        elif self._state == "lift_up_base":
            target_pos = self.prev_pose[:3, 3] + torch.tensor(
                [0, 0, 0.19], device=device
            )
            target_ori = ee_pose[:3, :3]
            target = self.add_noise_first_target(
                C.to_homogeneous(target_pos, target_ori)
            )
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.01, ori_error_threshold=0.3
            ):
                self.prev_pose = target
                next_state = "move_center"
        elif self._state == "move_center":
            target_pos = torch.tensor(
                [self.prev_pose[0, 3], 0.0, self.prev_pose[2, 3]], device=device
            )
            target_ori = self.prev_pose[:3, :3]
            target = self.add_noise_first_target(
                C.to_homogeneous(target_pos, target_ori)
            )
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.01, ori_error_threshold=0.3
            ):
                self.prev_pose = target
                next_state = "reach_base_xy"
        elif self._state == "reach_base_xy":
            base_screw_pose_robot = (
                april_to_robot
                @ leg_pose
                @ torch.tensor(
                    get_mat(self.default_assembled_pose[:3, 3], [0.0, 0.0, 0.0]),
                    device=device,
                )
            )
            base_screw_pose_robot[:3, :3] = self.prev_pose[:3, :3]
            target = base_screw_pose_robot
            # Keep the z.
            target[2, 3] = self.prev_pose[2, 3]
            if self.satisfy(
                ee_pose,
                target,
                pos_error_threshold=0.0,
                ori_error_threshold=0.0,
                max_len=50,
            ):
                self.prev_pose = target
                next_state = "reach_base_z"
        elif self._state == "reach_base_z":
            base_screw_pose_robot = (
                april_to_robot
                @ leg_pose
                @ torch.tensor(
                    get_mat(self.default_assembled_pose[:3, 3], [0.0, 0.0, 0.0]),
                    device=device,
                )
            )
            target = self.prev_pose
            target[2, 3] = base_screw_pose_robot[2, 3] + 0.020  # margin.
            base_screw_pose_robot[:3, :3] = self.prev_pose[:3, :3]
            target[:3, :3] = self.prev_pose[:3, :3]  # Keep the same orientation.
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.0, ori_error_threshold=0.0
            ):
                self.prev_pose = target
                next_state = "insert_base"
        elif self._state == "insert_base":  # Transition for skill labeling.
            target = self.prev_pose
            self.prev_pose = target
            next_state = "screw_gripper"
        elif self._state == "screw_gripper":
            target_pos = self.prev_pose[:3, 3]
            target_ori = (
                torch.tensor(rot_mat([np.pi, 0, -np.pi / 2])).to(device).float()
            )
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(
                ee_pose,
                target,
                pos_error_threshold=0.0,
                ori_error_threshold=0.0,
                max_len=30,
            ):
                self.prev_pose = target
                next_state = "release_screw_gripper"
        elif self._state == "release_screw_gripper":
            target = self.prev_pose
            self.gripper_action = -1
            if self.gripper_greater(
                gripper_width,
                config["robot"]["max_gripper_width"]["round_table"] - 0.001,
            ):
                self.prev_pose = target
                next_state = "go_up_screw"
        elif self._state == "go_up_screw":
            target_pos = self.prev_pose[:3, 3] + torch.tensor(
                [0, 0, 0.04], device=device
            )
            target_ori = ee_pose[:3, :3]
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.01, ori_error_threshold=0.3
            ):
                self.prev_pose = target
                next_state = "pre_grasp_front"
        elif self._state == "pre_grasp_front":
            # Face EE front.
            target = self.prev_pose
            target[:3, :3] = torch.tensor(rot_mat([np.pi, 0, 0])).to(device).float()
            if self.satisfy(ee_pose, target):
                self.prev_pose = target
                next_state = "pre_grasp_ori"
        elif self._state == "pre_grasp_ori":
            if base_pose[1, 1] < 0:
                rot = torch.tensor(
                    rot_mat([np.pi, 0, 0], hom=True), device=device
                ).float()
                theta_y = torch.acos(base_pose[1, 1]).detach().cpu().numpy()
                theta = np.pi - theta_y + np.pi / 4
                target_ori = (
                    torch.tensor(
                        rot_mat([0, 0, theta], hom=True), device=device
                    ).float()
                    @ rot
                )[:3, :3]
                pos = base_pose[:4, 3]
                target_pos = (april_to_robot @ pos)[:3]
                target_pos[2] = ee_pos[2]  # Keep the z.
                target = C.to_homogeneous(target_pos, target_ori)
            else:
                rot = torch.tensor(
                    rot_mat([np.pi, 0, 0], hom=True), device=device
                ).float()
                theta_y = torch.acos(base_pose[1, 1]).detach().cpu().numpy()
                theta = theta_y + np.pi / 4
                target_ori = (
                    torch.tensor(
                        rot_mat([0, 0, theta], hom=True), device=device
                    ).float()
                    @ rot
                )[:3, :3]
                pos = base_pose[:4, 3]
                target_pos = (april_to_robot @ pos)[:3]
                target_pos[2] = ee_pos[2]  # Keep the z.
                target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(ee_pose, target, pos_error_threshold=0.01):
                self.prev_pose = target.clone()
                next_state = "pre_grasp_z"
        elif self._state == "pre_grasp_z":
            if base_pose[1, 1] < 0:
                rot = torch.tensor(
                    rot_mat([np.pi, 0, 0], hom=True), device=device
                ).float()
                theta_y = torch.acos(base_pose[1, 1]).detach().cpu().numpy()
                theta = np.pi - theta_y + np.pi / 4
                target_ori = (
                    torch.tensor(
                        rot_mat([0, 0, theta], hom=True), device=device
                    ).float()
                    @ rot
                )[:3, :3]
                pos = base_pose[:4, 3]
                target_pos = (april_to_robot @ pos)[:3]
                target_pos[2] += 0.02  # Margin.
                target = C.to_homogeneous(target_pos, target_ori)
            else:
                rot = torch.tensor(
                    rot_mat([np.pi, 0, 0], hom=True), device=device
                ).float()
                theta_y = torch.acos(base_pose[1, 1]).detach().cpu().numpy()
                theta = theta_y + np.pi / 4
                target_ori = (
                    torch.tensor(
                        rot_mat([0, 0, theta], hom=True), device=device
                    ).float()
                    @ rot
                )[:3, :3]
                pos = base_pose[:4, 3]
                target_pos = (april_to_robot @ pos)[:3]
                target_pos[2] += 0.02  # Margin.
                target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(ee_pose, target, pos_error_threshold=0.005, max_len=40):
                self.prev_pose = target
                next_state = "screw_grasp"
        elif self._state == "screw_grasp":
            target = self.prev_pose
            self.gripper_action = 1
            if self.gripper_less(gripper_width, self.base_grip_width + 0.001):
                self.prev_pose = target
                next_state = "screw_front"
        elif self._state == "screw_front":
            # Face EE front.
            target = self.prev_pose
            target[:3, :3] = torch.tensor(
                get_mat([0, 0, 0], [np.pi, 0, 1 / 4 * np.pi])
            )[:3, :3]
            if self.satisfy(ee_pose, target, max_len=30):
                self.prev_pose = target
                next_state = "screw_gripper"
        skill_complete = self.may_transit_state(next_state)

        return (
            target[:3, 3],
            C.mat2quat(target[:3, :3]),
            torch.tensor([self.gripper_action], device=device),
            skill_complete,
        )
