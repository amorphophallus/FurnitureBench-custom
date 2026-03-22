import numpy as np
import torch

from furniture_bench.furniture.parts.part import Part
from furniture_bench.utils.pose import get_mat, rot_mat
import furniture_bench.utils.transform as T
import furniture_bench.controllers.control_utils as C
from furniture_bench.config import config


class LampHood(Part):
    def __init__(self, part_config, part_idx):
        super().__init__(part_config, part_idx)
        self.assembled_rel_poses = [
            get_mat([0, 0.094, 0], [0, 0, 0]),
            get_mat([0, 0.085, 0], [0, 0, 0]),
            get_mat([0, 0.09, 0], [0, 0, 0]),
        ]
        self.reset_x_len = 0.088
        self.reset_y_len = 0.088

        self.rel_pose_from_center[self.tag_ids[0]] = get_mat(
            [0, 0, -0.034022], [-0.139626, 0, 0]
        )
        self.rel_pose_from_center[self.tag_ids[1]] = get_mat(
            [np.cos(np.pi / 3) * 0.034022, 0, -np.sin(np.pi / 6) * 0.034022],
            [-0.139626, -np.pi / 3, 0],
        )
        self.rel_pose_from_center[self.tag_ids[2]] = get_mat(
            [np.cos(np.pi / 3) * 0.034022, 0, np.sin(np.pi / 6) * 0.034022],
            [-0.139626, -2 * np.pi / 3, 0],
        )
        self.rel_pose_from_center[self.tag_ids[3]] = get_mat(
            [0, 0, 0.034022], [-0.139626, -3 * np.pi / 3, 0]
        )
        self.rel_pose_from_center[self.tag_ids[4]] = get_mat(
            [np.cos(np.pi / 3) * 0.034022, 0, np.sin(np.pi / 6) * 0.034022],
            [-0.139626, -4 * np.pi / 3, 0],
        )
        self.rel_pose_from_center[self.tag_ids[5]] = get_mat(
            [np.cos(np.pi / 3) * 0.034022, 0, -np.sin(np.pi / 6) * 0.034022],
            [-0.139626, -5 * np.pi / 3, 0],
        )

        self.reset_gripper_width = 0.045
        self.init_ee_pos = None

        self.skill_complete_next_states = ["lift_up_hood"]
        self.hood_grip_width = 0.01
        self.pick_offset_y = 0.032
        self.place_height_offset = 0.07
        self.reset_skill_state()

    def is_in_reset_ori(self, pose, from_skill, ori_bound):
        return pose[2, 1] > 0.8

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

    def get_skill_label(self):
        if self.skill_state == "done":
            return None
        return self.skill_state

    def get_guidance_point(self):
        return self.skill_guidance_point

    def _compute_skill_pick_target(
        self,
        ee_pose,
        rb_states,
        part_idxs,
        sim_to_april_mat,
        april_to_robot,
    ):
        hood_pose = C.to_homogeneous(
            rb_states[part_idxs[self.name]][0][:3],
            C.quat2mat(rb_states[part_idxs[self.name]][0][3:7]),
        )
        hood_pose = sim_to_april_mat @ hood_pose
        target_pos = (april_to_robot @ hood_pose[:4, 3])[:3]
        target_pos[1] += self.pick_offset_y
        target_pos[2] = (april_to_robot @ hood_pose)[2, 3]
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
        base_pose = C.to_homogeneous(
            rb_states[part_idxs[assemble_to]][0][:3],
            C.quat2mat(rb_states[part_idxs[assemble_to]][0][3:7]),
        )
        hood_pose = C.to_homogeneous(
            rb_states[part_idxs[self.name]][0][:3],
            C.quat2mat(rb_states[part_idxs[self.name]][0][3:7]),
        )
        base_pose = sim_to_april_mat @ base_pose
        hood_pose = sim_to_april_mat @ hood_pose
        hood_pose_robot = april_to_robot @ hood_pose
        base_target_pose_robot = (
            april_to_robot
            @ base_pose
            @ torch.tensor(
                get_mat(self.default_assembled_pose[:3, 3], [0.0, 0.0, 0.0]),
                device=device,
            )
        )
        target_hood_pose_robot = hood_pose_robot.clone()
        target_hood_pose_robot[:3, 3] = base_target_pose_robot[:3, 3]
        target_hood_pose_robot[2, 3] += self.place_height_offset
        rel = target_hood_pose_robot @ torch.linalg.inv(hood_pose_robot)
        return rel @ ee_pose

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

        if self.skill_state == "pick":
            self.skill_guidance_point = self._compute_skill_pick_target(
                ee_pose,
                rb_states,
                part_idxs,
                sim_to_april_mat,
                april_to_robot,
            )
            pinched = False
            narrow_gripper = gripper_width < config["robot"]["max_gripper_width"]["lamp"] * 0.8
            if part_force is not None:
                part_force = part_force.clone()
                part_force = part_force - torch.dot(part_force, table_normal) * table_normal
                left_dist = torch.linalg.norm(
                    left_finger_pos - rb_states[part_idxs[self.name]][0][:3]
                )
                right_dist = torch.linalg.norm(
                    right_finger_pos - rb_states[part_idxs[self.name]][0][:3]
                )
                close_to_hood = left_dist < 0.1 and right_dist < 0.1
                pinched = torch.linalg.norm(part_force) > 1e-3 and close_to_hood and narrow_gripper
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
        hood_pose = C.to_homogeneous(
            rb_states[part_idxs["lamp_hood"]][0][:3],
            C.quat2mat(rb_states[part_idxs["lamp_hood"]][0][3:7]),
        )

        hood_pose = sim_to_april_mat @ hood_pose
        device = hood_pose.device

        if self.init_ee_pos is None:
            self.init_ee_pos = ee_pos.clone()

        if self._state == "move_up":
            self.gripper_action = -1
            target_pos = self.init_ee_pos + torch.tensor([0, 0, 0.08], device=device)
            target_ori = ee_pose[:3, :3]
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.00, ori_error_threshold=0.0
            ):
                self.prev_pose = target
                next_state = "move_center"
        elif self._state == "move_center":
            z = ee_pose[2, 3]
            target_pos = torch.tensor([self.prev_pose[0, 3], -0.05, z], device=device)
            target_ori = self.prev_pose[:3, :3]
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.02, ori_error_threshold=0.3
            ):
                self.prev_pose = target
                next_state = "reach_hood_floor_xy"
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
        base_pose = C.to_homogeneous(
            rb_states[part_idxs[assemble_to]][0][:3],
            C.quat2mat(rb_states[part_idxs[assemble_to]][0][3:7]),
        )
        hood_pose = C.to_homogeneous(
            rb_states[part_idxs[self.name]][0][:3],
            C.quat2mat(rb_states[part_idxs[self.name]][0][3:7]),
        )

        hood_pose = sim_to_april_mat @ hood_pose
        base_pose = sim_to_april_mat @ base_pose

        device = ee_pose.device

        if self._state == "reach_hood_floor_xy":
            target_ori = (torch.tensor(rot_mat([np.pi, 0, 0], hom=True)).float())[
                :3, :3
            ]
            pos = hood_pose[:4, 3]
            target_pos = (april_to_robot @ pos)[:3]
            target_pos[2] = ee_pos[2]

            target_pos[1] += 0.032
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(ee_pose, target, pos_error_threshold=0.02):
                self.prev_pose = target.clone()
                next_state = "reach_hood_floor_z"
        elif self._state == "reach_hood_floor_z":
            target_pos = self.prev_pose[:3, 3]
            target_pos[2] = (april_to_robot @ hood_pose)[2, 3]
            target_ori = self.prev_pose[:3, :3]
            target = C.to_homogeneous(target_pos, target_ori)
            target[2, 3] += 0.04
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.01, ori_error_threshold=0.3
            ):
                self.prev_pose = target
                next_state = "pick_hood"
        elif self._state == "pick_hood":
            target = self.prev_pose
            self.gripper_action = 1
            if self.gripper_less(
                gripper_width, self.hood_grip_width + 0.001, cnt_max=20
            ):
                self.prev_pose = target
                next_state = "lift_up_hood"
        elif self._state == "lift_up_hood":
            target_pos = self.prev_pose[:3, 3] + torch.tensor(
                [0, 0, 0.19], device=device
            )
            target_ori = ee_pose[:3, :3]
            target = self.add_noise_first_target(
                C.to_homogeneous(target_pos, target_ori)
            )
            if self.satisfy(
                ee_pose,
                target,
                pos_error_threshold=0.01,
                ori_error_threshold=0.3,
                max_len=35,
            ):
                self.prev_pose = target
                next_state = "move_center"
        elif self._state == "move_center":
            target_pos = torch.tensor([self.prev_pose[0, 3], 0.0, self.prev_pose[2, 3]], device=device)
            target_ori = self.prev_pose[:3, :3]
            target = self.add_noise_first_target(
                C.to_homogeneous(target_pos, target_ori)
            )
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.01, ori_error_threshold=0.3
            ):
                self.prev_pose = target
                next_state = "reach_base_top_xy"
        elif self._state == "reach_base_top_xy":
            base_pose_robot = (
                april_to_robot
                @ base_pose
                @ torch.tensor(
                    get_mat(self.default_assembled_pose[:3, 3], [0.0, 0.0, 0.0]),
                    device=device,
                )
            )
            base_pose_robot[:3, :3] = self.prev_pose[:3, :3]
            target = base_pose_robot.clone()
            target[1, 3] += 0.032
            target[2, 3] = self.prev_pose[2, 3]
            if self.satisfy(
                ee_pose,
                target,
                pos_error_threshold=0.0,
                ori_error_threshold=0.0,
                max_len=50,
            ):
                self.prev_pose = target
                next_state = "reach_base_top_z"
        elif self._state == "reach_base_top_z":
            base_pose_robot = (
                april_to_robot
                @ base_pose
                @ torch.tensor(
                    get_mat(self.default_assembled_pose[:3, 3], [0.0, 0.0, 0.0]),
                    device=device,
                )
            )
            target = self.prev_pose
            target[2, 3] = base_pose_robot[2, 3] + 0.07
            base_pose_robot[:3, :3] = self.prev_pose[:3, :3]
            target[:3, :3] = self.prev_pose[:3, :3]
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.0, ori_error_threshold=0.0
            ):
                self.prev_pose = target
                next_state = "release_gripper"
        elif self._state == "release_gripper":
            target = self.prev_pose
            self.gripper_action = -1
            if self.gripper_greater(
                gripper_width,
                config["robot"]["max_gripper_width"]["cabinet"] - 0.001,
            ):
                self.prev_pose = target
                next_state = "done"
        elif self._state == "done":
            target = self.prev_pose
            self.gripper_action = -1
        skill_complete = self.may_transit_state(next_state)

        return (
            target[:3, 3],
            C.mat2quat(target[:3, :3]),
            torch.tensor([self.gripper_action], device=device),
            skill_complete,
        )
