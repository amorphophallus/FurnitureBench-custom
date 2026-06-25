import torch
import numpy as np
import numpy.typing as npt

from furniture_bench.furniture.parts.part import Part
from furniture_bench.utils.pose import get_mat, is_similar_rot, is_similar_xz, rot_mat
from furniture_bench.config import config
import furniture_bench.utils.transform as T
import furniture_bench.controllers.control_utils as C


class Leg(Part):
    def __init__(self, part_config, part_idx):
        super().__init__(part_config, part_idx)
        tag_ids = part_config["ids"]

        self.rel_pose_from_center[tag_ids[0]] = get_mat(
            [0, 0, -self.tag_offset], [0, 0, 0]
        )
        self.rel_pose_from_center[tag_ids[1]] = get_mat(
            [-self.tag_offset, 0, 0], [0, np.pi / 2, 0]
        )
        self.rel_pose_from_center[tag_ids[2]] = get_mat(
            [0, 0, self.tag_offset], [0, np.pi, 0]
        )
        self.rel_pose_from_center[tag_ids[3]] = get_mat(
            [self.tag_offset, 0, 0], [0, -np.pi / 2, 0]
        )

        self.done = False
        self.pos_error_threshold = 0.01
        self.ori_error_threshold = 0.25

        self.skill_complete_next_states = [
            "lift_up",
            "insert",
        ]  # Specificy next state after skill is complete. Screw done is handle in `get_assembly_action`

        self.reset()

        self.part_attached_skill_idx = 4

    def reset(self):
        self.prev_pose = None
        self._state = "reach_leg_floor_xy"
        self.gripper_action = -1
        self.reset_skill_state()

    def reset_skill_state(self):
        self.skill_state = "pick"
        self.skill_target_ee_pose_robot = None
        self.skill_target_part_pose_robot = None
        self.skill_target_anchor_pose_robot = None
        self.skill_guidance_point_robot = None
        self.skill_pinched_steps = 0
        self.skill_place_xy_threshold = 0.008
        self.skill_place_z_threshold = 0.04
        self.skill_place_ori_threshold = 0.5
        self.skill_place_part_xz_threshold = 0.007
        self.skill_place_part_ori_threshold = 0.15

    def get_skill_label(self):
        if self.skill_state == "done":
            return None
        return self.skill_state

    def get_guidance_point(self):
        return self.skill_guidance_point_robot

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

    def _rot_error_ignore_axis(self, current_rot, target_rot, ignore_axis):
        rel_rot = current_rot @ target_rot.T
        trace = torch.trace(rel_rot)
        cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
        angle = torch.acos(cos_angle)
        if torch.isclose(angle, torch.tensor(0.0, device=current_rot.device)):
            axis_angle = torch.zeros(3, device=current_rot.device)
        else:
            axis = torch.stack(
                [
                    rel_rot[2, 1] - rel_rot[1, 2],
                    rel_rot[0, 2] - rel_rot[2, 0],
                    rel_rot[1, 0] - rel_rot[0, 1],
                ]
            ) / (2.0 * torch.sin(angle))
            axis_angle = axis * angle
        axis_angle_ignore = axis_angle.clone()
        axis_angle_ignore[ignore_axis] = 0.0
        return torch.linalg.norm(axis_angle_ignore)

    def _part_pose_env(self, part_name, rb_states, part_idxs):
        return C.to_homogeneous(
            rb_states[part_idxs[part_name]][0][:3],
            C.quat2mat(rb_states[part_idxs[part_name]][0][3:7]),
        )

    def _part_pose_robot(self, part_name, rb_states, part_idxs, sim_to_april_mat, april_to_robot):
        part_pose_env = self._part_pose_env(part_name, rb_states, part_idxs)
        part_pose_april = sim_to_april_mat @ part_pose_env
        return april_to_robot @ part_pose_april

    def _part_place_errors_robot(
        self,
        part_pose_robot,
        anchor_pose_robot,
        *,
        ignore_axis,
        planar_axes=(0, 2),
        vertical_axis=1,
    ):
        current_rel_part_pose_robot = torch.linalg.inv(anchor_pose_robot) @ part_pose_robot
        target_rel_part_pose_robot = (
            torch.linalg.inv(self.skill_target_anchor_pose_robot)
            @ self.skill_target_part_pose_robot
        )
        part_pos_diff_robot = (
            current_rel_part_pose_robot[:3, 3] - target_rel_part_pose_robot[:3, 3]
        )
        planar_error = part_pos_diff_robot[list(planar_axes)].abs().sum()
        vertical_error = part_pos_diff_robot[vertical_axis].abs()
        ori_error = self._rot_error_ignore_axis(
            current_rel_part_pose_robot[:3, :3],
            target_rel_part_pose_robot[:3, :3],
            ignore_axis=ignore_axis,
        )
        return planar_error, vertical_error, ori_error

    def _compute_skill_pick_target(
        self,
        ee_pose,
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
        leg_pose_april = self._find_down_z(leg_pose_april).clone().to(ee_pose.device)
        pos_april = leg_pose_april[:4, 3]
        target_pos_robot = (april_to_robot @ pos_april)[:3]
        target_pos_robot[1] += 0.01
        target_pos_robot[0] += self.grasp_margin_x
        target_pos_robot[2] = (april_to_robot @ leg_pose_april)[2, 3]
        return target_pos_robot

    def _compute_skill_screw_target(
        self,
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
        leg_pose_robot = april_to_robot @ leg_pose_april
        # Offset 1/4 leg length along local Y (long axis)
        y_offset_robot = leg_pose_robot[:3, 1] * (self.reset_y_len * 0.25)
        return (leg_pose_robot[:3, 3] + y_offset_robot).clone()

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
        # rb_states positions are in env-local frame
        table_pose_env = C.to_homogeneous(
            rb_states[part_idxs[assemble_to]][0][:3],
            C.quat2mat(rb_states[part_idxs[assemble_to]][0][3:7]),
        )
        leg_pose_env = C.to_homogeneous(
            rb_states[part_idxs[self.name]][0][:3],
            C.quat2mat(rb_states[part_idxs[self.name]][0][3:7]),
        )
        table_pose_april = sim_to_april_mat @ table_pose_env
        leg_pose_april = sim_to_april_mat @ leg_pose_env
        leg_pose_robot = april_to_robot @ leg_pose_april
        table_pose_robot = april_to_robot @ table_pose_april
        table_hole_pose_robot = (
            table_pose_robot
            @ torch.tensor(
                get_mat(self.default_assembled_pose[:3, 3], [0.0, 0.0, 0.0]),
                device=device,
            )
        )
        target_leg_pose_robot = torch.tensor(
            [
                [1.0, 0.0, 0.0, table_hole_pose_robot[0, 3]],
                [0.0, 0.0, -1.0, table_hole_pose_robot[1, 3]],
                [0.0, 1.0, 0.0, table_pose_robot[2, 3] + 0.06],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
        )
        rel_robot = target_leg_pose_robot @ torch.linalg.inv(leg_pose_robot)
        target_ee_pose_robot = rel_robot @ ee_pose_robot
        self.skill_target_part_pose_robot = target_leg_pose_robot.clone()
        self.skill_target_anchor_pose_robot = table_pose_robot.clone()
        return target_ee_pose_robot

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

        if self.skill_state == "pick":
            self.skill_guidance_point_robot = self._compute_skill_pick_target(
                ee_pose_robot,
                rb_states,
                part_idxs,
                sim_to_april_mat,
                april_to_robot,
            )
            pinched = False
            leg_force_mag = None
            left_dist = None
            right_dist = None
            close_to_leg = False
            narrow_gripper = (
                gripper_width
                < config["robot"]["max_gripper_width"]["square_table"] * 0.8
            )
            if part_force is not None:
                part_force = part_force.clone()
                part_force = part_force - torch.dot(part_force, table_normal) * table_normal
                leg_force_mag = torch.linalg.norm(part_force)
                left_dist = torch.linalg.norm(left_finger_pos - rb_states[part_idxs[self.name]][0][:3])
                right_dist = torch.linalg.norm(right_finger_pos - rb_states[part_idxs[self.name]][0][:3])
                close_to_leg = left_dist < 0.06 and right_dist < 0.06
                pinched = leg_force_mag > 1e-3 and close_to_leg and narrow_gripper
            # print(
            #     "[leg pick_debug] "
            #     f"force_mag={(leg_force_mag.item() if leg_force_mag is not None else None)} "
            #     "force_thresh=0.001 "
            #     f"force_ok={(leg_force_mag is not None and leg_force_mag.item() > 1e-3)} "
            #     f"left_dist={(left_dist.item() if left_dist is not None else None)} "
            #     f"right_dist={(right_dist.item() if right_dist is not None else None)} "
            #     "dist_thresh=0.06 "
            #     f"close_ok={close_to_leg} "
            #     f"gripper_width={gripper_width.item():.6f} "
            #     f"gripper_thresh={(config['robot']['max_gripper_width']['square_table'] * 0.8):.6f} "
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
            table_pose_env = C.to_homogeneous(
                rb_states[part_idxs[assemble_to]][0][:3],
                C.quat2mat(rb_states[part_idxs[assemble_to]][0][3:7]),
            )
            leg_pose_env = C.to_homogeneous(
                rb_states[part_idxs[self.name]][0][:3],
                C.quat2mat(rb_states[part_idxs[self.name]][0][3:7]),
            )
            table_pose_robot = april_to_robot @ sim_to_april_mat @ table_pose_env
            leg_pose_robot = april_to_robot @ sim_to_april_mat @ leg_pose_env
            xy_error, z_error, ori_error = self._part_place_errors_robot(
                leg_pose_robot,
                table_pose_robot,
                ignore_axis=1,
            )
            if (
                xy_error < self.skill_place_part_xz_threshold
                and z_error < self.skill_place_z_threshold
                and ori_error < self.skill_place_part_ori_threshold
            ):
                self.skill_state = "insert"
        elif self.skill_state == "insert":
            self.skill_target_ee_pose_robot = self._compute_skill_place_target(
                ee_pose_robot,
                rb_states,
                part_idxs,
                sim_to_april_mat,
                april_to_robot,
                assemble_to,
            )
            self.skill_guidance_point_robot = self.skill_target_ee_pose_robot[:3, 3].clone()
            if gripper_width >= config["robot"]["max_gripper_width"]["square_table"] - 0.001:
                self.skill_state = "screw"
                self.skill_guidance_point_robot = self._compute_skill_screw_target(
                    rb_states,
                    part_idxs,
                    sim_to_april_mat,
                    april_to_robot,
                )
        elif self.skill_state == "screw":
            self.skill_guidance_point_robot = self._compute_skill_screw_target(
                rb_states,
                part_idxs,
                sim_to_april_mat,
                april_to_robot,
            )
            if assembled:
                self.skill_state = "done"

        return self.skill_state

    def is_in_reset_ori(
        self, pose: npt.NDArray[np.float32], from_skill, ori_bound
    ) -> bool:
        # y-axis of the leg align with y-axis of the base.
        reset_ori = (
            self.reset_ori[from_skill] if len(self.reset_ori) > 1 else self.reset_ori[0]
        )
        for _ in range(4):
            if is_similar_rot(pose[:3, :3], reset_ori[:3, :3], ori_bound=ori_bound):
                return True
            pose = pose @ rot_mat(np.array([0, np.pi / 2, 0]), hom=True)
        return False

    def _find_down_z(self, mat):
        for _ in range(4):
            if mat[2, 2] > 0.8:  # Z is down.
                break
            mat = mat @ torch.tensor(
                T.rotmat2hom(rot_mat([0, np.pi / 2, 0]))
            ).float().to(mat.device)
        return mat

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
        table_pose = C.to_homogeneous(
            rb_states[part_idxs[assemble_to]][0][:3],
            C.quat2mat(rb_states[part_idxs[assemble_to]][0][3:7]),
        )
        leg_pose = C.to_homogeneous(
            rb_states[part_idxs[self.name]][0][:3],
            C.quat2mat(rb_states[part_idxs[self.name]][0][3:7]),
        )

        table_pose = sim_to_april_mat @ table_pose
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
            rot = rot_mat_tensor(np.pi / 2, -np.pi / 2, 0, device)

            pos = leg_pose[:4, 3]

            target_pos = (april_to_robot @ pos)[:3]
            # target_ori = (margin @ april_to_robot @ rot)[:3, :3]
            target_ori = ee_pose[:3, :3]
            target_pos[2] = ee_pos[2]
            target_pos[1] += 0.01
            target_pos[0] += self.grasp_margin_x
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
            target_ori = (margin @ april_to_robot @ rot)[:3, :3]
            target_pos = self.prev_pose[:3, 3]
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(ee_pose, target, pos_error_threshold=0.015):
                self.prev_pose = target
                next_state = "reach_leg_floor_z"
        elif self._state == "reach_leg_floor_z":
            target_pos = self.prev_pose[:3, 3]
            target_pos[2] = (april_to_robot @ leg_pose)[2, 3]
            target_ori = self.prev_pose[:3, :3]
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
                [0, 0, 0.10], device=device
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
            target_pos = torch.tensor([0.5, 0.1, 0.1], device=device)
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
            target_ori = (margin @ rot_mat_tensor(np.pi, 0, 0, device))[:3, :3]
            target_pos = torch.tensor([0.57, 0.1, 0.12], device=device)
            target = self.add_noise_first_target(
                C.to_homogeneous(target_pos, target_ori)
            )
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.02, ori_error_threshold=0.3
            ):
                self.prev_pose = target
                next_state = "reach_table_top_xy"
        elif self._state == "reach_table_top_xy":
            leg_pose_robot = april_to_robot @ leg_pose
            leg_pose_robot = find_leg_pose_x_look_front(leg_pose_robot)
            table_hole_pose_robot = (
                april_to_robot
                @ table_pose
                @ torch.tensor(
                    get_mat(self.default_assembled_pose[:3, 3], [0.0, 0.0, 0.0]),
                    device=device,
                )
            )
            target_leg_pose_robot = torch.tensor(
                [
                    [1.0, 0.0, 0.0, table_hole_pose_robot[0, 3]],
                    [0.0, 0.0, -1.0, table_hole_pose_robot[1, 3]],
                    [0.0, 1.0, 0.0, self.prev_pose[2, 3]],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                device=device,
            )
            rel = rel_rot_mat(leg_pose_robot, target_leg_pose_robot)
            target = rel @ ee_pose
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.015, ori_error_threshold=0.3
            ):
                self.prev_pose = target
                next_state = "reach_table_top_z"
        elif self._state == "reach_table_top_z":
            leg_pose_robot = april_to_robot @ leg_pose
            leg_pose_robot = find_leg_pose_x_look_front(leg_pose_robot)
            table_hole_pose_robot = (
                april_to_robot
                @ table_pose
                @ torch.tensor(
                    get_mat(self.default_assembled_pose[:3, 3], [0.0, 0.0, 0.0]),
                    device=device,
                )
            )
            target_leg_pose_robot = torch.tensor(
                [
                    [1.0, 0.0, 0.0, table_hole_pose_robot[0, 3]],
                    [0.0, 0.0, -1.0, table_hole_pose_robot[1, 3]],
                    [0.0, 1.0, 0.0, table_pose[2, 3] + 0.09],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                device=device,
            )
            rel = rel_rot_mat(leg_pose_robot, target_leg_pose_robot)
            target = rel @ ee_pose
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.007, ori_error_threshold=0.15
            ):
                self.prev_pose = target
                next_state = "insert_wait"
        elif self._state == "insert_wait":
            leg_pose_robot = april_to_robot @ leg_pose
            leg_pose_robot = find_leg_pose_x_look_front(leg_pose_robot)
            table_hole_pose_robot = (
                april_to_robot
                @ table_pose
                @ torch.tensor(
                    get_mat(self.default_assembled_pose[:3, 3], [0.0, 0.0, 0.0]),
                    device=device,
                )
            )
            target_leg_pose_robot = torch.tensor(
                [
                    [1.0, 0.0, 0.0, table_hole_pose_robot[0, 3]],
                    [0.0, 0.0, -1.0, table_hole_pose_robot[1, 3]],
                    [0.0, 1.0, 0.0, table_pose[2, 3] + 0.084],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                device=device,
            )
            rel = rel_rot_mat(leg_pose_robot, target_leg_pose_robot)
            target = rel @ ee_pose
            if self.satisfy(
                ee_pose,
                target,
                pos_error_threshold=0.0,
                ori_error_threshold=0.0,
                max_len=10,
            ):
                self.prev_pose = target
                next_state = "insert_release"
        elif self._state == "insert_release":
            target = self.prev_pose
            self.gripper_action = -1
            if self.gripper_greater(
                gripper_width,
                config["robot"]["max_gripper_width"]["square_table"] - 0.001,
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
                config["robot"]["max_gripper_width"]["square_table"] - 0.001,
            ):
                next_state = "pre_grasp"
        elif self._state == "pre_grasp":
            target_ori = rot_mat_tensor(np.pi, 0, 0, device)[:3, :3]
            target_pos = (april_to_robot @ leg_pose[:4, 3])[:3]
            target_pos[2] += self.grasp_margin_z
            # target = self.add_noise_first_target(C.to_homogeneous(target_pos, target_ori))
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
            # 'screw_grasp', 'screw', 'match_leg_ori', 'reach_table_top_xy', 'reach_table_top_z'
            "insert_wait",
            "insert_release",
            "insert",
        ]

    def _find_closest_y(self, pose):
        closest_y = pose.clone()
        for i in range(4):
            tmp_pose = pose @ torch.tensor(
                self.rel_pose_from_center[self.tag_ids[i]]
            ).float().to(pose.device)
            if tmp_pose[1, 3] < closest_y[1, 3]:
                closest_y = tmp_pose
        return closest_y
