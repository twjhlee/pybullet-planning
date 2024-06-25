#!/usr/bin/env python3

from __future__ import print_function

from pybullet_tools.franka_primitives import BodyPose, BodyConf, Command, get_grasp_gen, \
    get_ik_fn, get_free_motion_gen, get_holding_motion_gen, get_grasping_fn, BodyGrasp
from pybullet_tools.utils import WorldSaver, enable_gravity, connect, dump_world, set_pose, \
    draw_global_system, Pose, Point, set_default_camera, BLOCK_URDF, load_model, wait_if_gui, \
    disconnect, DRAKE_IIWA_URDF, update_state, disable_real_time, HideOutput, load_pybullet, \
    get_movable_joints, set_joint_positions, create_box, RGBA, get_point, get_euler, refine_path, \
    inverse_kinematics, end_effector_from_body, get_pose, link_from_name, get_link_pose, invert

from pybullet_tools.ikfast.franka_panda.ik import FRANKA_URDF, PANDA_INFO

import os
import pybullet as p
import numpy as np
import argparse
from glob import glob
import pickle

from scipy.spatial.transform import Rotation as R

def plan(robot, block, fixed, teleport):
    grasp_gen = get_grasp_gen(robot, 'top')
    ik_fn = get_ik_fn(robot, fixed=fixed, teleport=teleport, num_attempts=100)
    free_motion_fn = get_free_motion_gen(robot, fixed=([block] + fixed), teleport=teleport)
    holding_motion_fn = get_holding_motion_gen(robot, fixed=fixed, teleport=teleport)

    pose0 = BodyPose(block)
    conf0 = BodyConf(robot)
    saved_world = WorldSaver()
    for grasp, in grasp_gen(block):
        saved_world.restore()
        result1 = ik_fn(block, pose0, grasp)
        if result1 is None:
            continue
        conf1, path2 = result1
        pose0.assign()
        result2 = free_motion_fn(conf0, conf1)
        if result2 is None:
            continue
        path1, = result2
        result3 = holding_motion_fn(conf1, conf0, block, grasp)
        if result3 is None:
            continue
        path3, = result3
        # path1.body_paths[0].path[0] -> joint angles
        return Command(path1.body_paths +
                          path2.body_paths +
                          path3.body_paths)
    return None

def plan_nft(robot, target_points, target_mesh, scene_mesh, floor, teleport, smoothing=False, algorithm=None):
    grasp_gen = get_grasp_gen(robot, 'perpendicular')
    grasping_fn = get_grasping_fn(robot, fixed=scene_mesh + [floor], teleport=teleport, num_attempts=10)
    free_motion_fn = get_free_motion_gen(robot, fixed=scene_mesh + [floor], teleport=teleport, smoothing=smoothing, algorithm=algorithm)
    holding_motion_fn = get_holding_motion_gen(robot, fixed=scene_mesh + [floor], teleport=teleport, algorithm=algorithm)

    target_poses = []
    for target in target_points:
        target_poses.append(BodyPose(target))

    init_pos = BodyConf(robot)

    saved_world = WorldSaver()

    for target_pose, target_point in zip(target_poses, target_points):
        for grasp, in grasp_gen(target_point):
            saved_world.restore()

            q_grasp = inverse_kinematics(robot, grasp.link,
                end_effector_from_body(target_pose.pose, grasp.grasp_pose))
            conf = BodyConf(robot, q_grasp)
            link_pose = get_link_pose(robot, link_from_name(robot, 'panda_hand'))

            in_motion_result = free_motion_fn(init_pos, conf) # move from initial position to grasp position
            if in_motion_result is None:
                continue
            in_motion_cmd, = in_motion_result
            out_motion_result = free_motion_fn(conf, init_pos)
            if out_motion_result is None:
                continue
            out_motion_cmd, = out_motion_result

            in_motion_cmd = in_motion_cmd.refine(num_steps=50)
            out_motion_cmd = out_motion_cmd.refine(num_steps=50)
            return (in_motion_cmd, out_motion_cmd)

    return None

def perpendicular_euler_angles(euler_angles):
    """
    Generates a set of Euler angles that represents a rotation perpendicular to the input rotation.
    """
    # Convert the input Euler angles to a rotation matrix
    r = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()

    # Compute the direction perpendicular to the input z-axis
    z_dir = np.array([0, 0, 1])
    perp_dir = np.cross(z_dir, r[:,2])

    # Generate a random angle of rotation
    angle = np.random.uniform(0, 2*np.pi)

    # Rotate the perpendicular direction about the input z-axis by the random angle
    rot_axis = r[:,2]
    rot = R.from_rotvec(angle * rot_axis)
    perp_dir_rotated = rot.apply(perp_dir)

    # Convert the rotated direction to Euler angles
    yaw = np.arctan2(perp_dir_rotated[1], perp_dir_rotated[0])
    pitch = np.arctan2(np.sqrt(perp_dir_rotated[0]**2 + perp_dir_rotated[1]**2), perp_dir_rotated[2])
    roll = 0.0

    # Convert the new Euler angles to degrees and return them
    return np.array([roll, pitch, yaw]) * 180 / np.pi

def get_orthogonal_vector(direction):
    """
    Finds an orthogonal vector to the given directional vector using the cross product.
    """
    # Choose a random vector to cross with the input vector
    rand_vec = np.random.randn(3)

    # Take the cross product of the input vector and the random vector
    ortho_vec = np.cross(direction, rand_vec)

    # If the cross product is zero, choose a different random vector
    while np.linalg.norm(ortho_vec) < 1e-6:
        rand_vec = np.random.randn(3)
        ortho_vec = np.cross(direction, rand_vec)

    # Normalize the orthogonal vector and return it
    return ortho_vec / np.linalg.norm(ortho_vec)

def vector_to_euler(direction):
    """
    Converts a directional vector to Euler angles.
    """
    yaw = np.arctan2(direction[1], direction[0])
    pitch = np.arctan2(direction[2], np.sqrt(direction[0]**2 + direction[1]**2))
    roll = 0.0  # There is no unique solution for roll

    return np.array([yaw, pitch, roll]).tolist()


def generate_grasp_targets(line):
    pass

def main(args, display='execute'): # control | execute | step

    connect(use_gui=True)
    disable_real_time()
    draw_global_system()
    # Read the lines pickle file.
    with open(os.path.join(args.path, '../Interact/lines.pickle'), 'rb') as f:
        lines = pickle.load(f)
    with HideOutput():
        # robot = load_model(FRANKA_URDF) # KUKA_IIWA_URDF | DRAKE_IIWA_URDF
        # robot = load_pybullet(FRANKA_URDF, fixed_base=True)
        robot = load_pybullet(FRANKA_URDF.replace('panda_arm_hand', 'panda_arm_hand_cam_closed'), fixed_base=True)
        # maybe change this?
        floor = load_model('models/short_floor.urdf')
        # read all urdfs of bbjects
        all_urdfs = sorted(glob(os.path.join(args.path, "mesh*.urdf")))
        scene_mesh = []
        for urdf_path in all_urdfs:
            mesh = p.loadURDF(urdf_path)
            scene_mesh.append(mesh)
    # Set the floor position
    set_pose(floor, Pose(Point(x=1.2, z=0.025)))

    joints = get_movable_joints(robot)

    set_default_camera(distance=2)
    dump_world()

    saved_world = WorldSaver()
    conf = (0, -np.pi / 4.0, 0, -3.0 * np.pi / 4.0, 0, np.pi / 2, np.pi / 4, 0.04, 0.04)
    # move robot to start position
    set_joint_positions(robot, joints, conf)

    # generate grasp targets
    grasp_targets = generate_grasp_targets(lines)

    # We should do the planning here
    while True:
        update_state()

    print('Quit?')
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="directory to urdfs")
    args = parser.parse_args()
    main(args, 'execute')
