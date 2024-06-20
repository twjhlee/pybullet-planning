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

import pybullet as p
import numpy as np

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

def main(display='execute'): # control | execute | step
    # grasp_points = np.load('models/grasp_points.npy')
    grasp_data = np.load('models/grasp_data.npz')
    x1_arr = grasp_data['x1']
    x2_arr = grasp_data['x2']
    target_cluster_arr = grasp_data['target_cluster']
    total_cluster_cnt = grasp_data['total_cluster_cnt']
    center_arr = ((x1_arr + x2_arr) / 2).tolist()
    direction_arr = x1_arr - x2_arr
    direction_arr = direction_arr / np.linalg.norm(direction_arr)

    grasp_targets = []
    perp_targets = []

    connect(use_gui=True)
    disable_real_time()
    draw_global_system()
    with HideOutput():
        # robot = load_model(FRANKA_URDF) # KUKA_IIWA_URDF | DRAKE_IIWA_URDF
        # robot = load_pybullet(FRANKA_URDF, fixed_base=True)
        robot = load_pybullet(FRANKA_URDF.replace('panda_arm_hand', 'panda_arm_hand_cam'), fixed_base=True)
        floor = load_model('models/short_floor.urdf')
        scene_mesh = []
        for i in range(total_cluster_cnt):
            mesh = p.loadURDF('models/scene_mesh_{}.urdf'.format(i))
            # if i == target_cluster:
            #     target_mesh = mesh
            # else:
            #     scene_mesh.append(mesh)
            scene_mesh.append(mesh)

        for _ in range(len(x1_arr)):
            grasp_target = create_box(w=0.01, l=0.01, h=0.01, color=RGBA(0, 1, 0, 0)) # Creates a red box obstacle
            perp_target = create_box(w=0.01, l=0.01, h=0.01, color=RGBA(1, 0, 0, 0)) # Creates a red box obstacle
            grasp_targets.append(grasp_target)
            perp_targets.append(perp_target)

        # set_point(obstacle, [0.5, 0.5, 0.1 / 2.]) # Sets the [x,y,z] position of the obstacle
        # print('Position:', get_point(obstacle))
        # set_euler(obstacle, [0, 0, np.pi / 4]) #  Sets the [roll,pitch,yaw] orientation of the obstacle
        # print('Orientation:', get_euler(obstacle))
    set_pose(floor, Pose(Point(x=1.2, z=0.025)))
    # set_pose(grasp_target, Pose(Point(x=0.39, y=0.03, z=0.15), [np.pi / 2, np.pi / 2, np.pi / 2]))

    for grasp_target, perp_target, center, direction in zip(grasp_targets, perp_targets, center_arr, direction_arr):
        set_pose(grasp_target, Pose(Point(x=center[0], y=center[1], z=center[2]), vector_to_euler(direction)))

        input_r = R.from_euler('xyz', get_euler(grasp_target), degrees=True).as_matrix()
        input_dir = input_r[:,2]

        ortho_dir = get_orthogonal_vector(input_dir)
        ortho_euler = vector_to_euler(ortho_dir)

        set_pose(perp_target, Pose(get_point(grasp_target), ortho_euler))

    joints = get_movable_joints(robot)

    set_default_camera(distance=2)
    dump_world()

    saved_world = WorldSaver()
    conf = (0, -np.pi / 4.0, 0, -3.0 * np.pi / 4.0, 0, np.pi / 2, np.pi / 4, 0.04, 0.04)
    set_joint_positions(robot, joints, conf)

    # command = plan(robot, object, fixed=[], teleport=False)
    # in_motion_cmd, leaving_cmd, out_motion_cmd = plan_nft(robot, target=perp_target, scene_mesh=scene_mesh, floor=floor, teleport=False, smoothing=False, algorithm='direct')
    in_motion_cmd, out_motion_cmd = plan_nft(robot,
                                             target_points=perp_targets,
                                             target_mesh=None,
                                             scene_mesh=scene_mesh,
                                             floor=floor,
                                             teleport=False,
                                             smoothing=False,
                                             algorithm='direct')
    command = Command(in_motion_cmd.body_paths +
                    #   leaving_cmd.body_paths +
                      out_motion_cmd.body_paths)
    in_motion_path = [pos for path in in_motion_cmd.body_paths for pos in path.path]
    in_motion_path = np.array(in_motion_path)
    # out_motion_path = np.array([pos for path in leaving_cmd.body_paths + out_motion_cmd.body_paths for pos in path.path])
    out_motion_path = [pos for path in out_motion_cmd.body_paths for pos in path.path]
    out_motion_path = np.array(out_motion_path)
    np.savetxt('/home/panda/libfranka/examples/trajectories/in_motion.txt', in_motion_path)
    np.savetxt('/home/panda/libfranka/examples/trajectories/out_motion.txt', out_motion_path)
    if (command is None) or (display is None):
        print('Unable to find a plan!')
        return

    saved_world.restore()
    update_state()
    wait_if_gui('{}?'.format(display))
    if display == 'control':
        enable_gravity()
        command.control(real_time=False, dt=0)
    elif display == 'execute':
        command.execute(time_step=0.002)
    elif display == 'step':
        command.step()
    else:
        raise ValueError(display)

    print('Quit?')
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main('execute')
