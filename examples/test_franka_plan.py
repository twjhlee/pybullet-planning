#!/usr/bin/env python

from __future__ import print_function

import pybullet as p
import numpy as np

from pybullet_tools.utils import add_data_path, connect, dump_body, disconnect, wait_for_user, \
    get_movable_joints, get_sample_fn, set_joint_positions, get_joint_name, LockRenderer, link_from_name, get_link_pose, \
    multiply, Pose, Point, interpolate_poses, HideOutput, draw_pose, set_camera_pose, load_pybullet, \
    assign_link_colors, add_line, point_from_pose, remove_handles, BLUE, INF, BLOCK_URDF, load_model, \
        set_pose, stable_z, RGBA, create_box, get_point, set_point, set_euler, get_euler, plan_direct_joint_motion

from pybullet_tools.ikfast.franka_panda.ik import PANDA_INFO, FRANKA_URDF
from pybullet_tools.ikfast.ikfast import get_ik_joints, either_inverse_kinematics, check_ik_solver

def test_retraction(robot, info, tool_link, distance=0.1, **kwargs):
    ik_joints = get_ik_joints(robot, info, tool_link)
    start_pose = get_link_pose(robot, tool_link)
    end_pose = multiply(start_pose, Pose(Point(z=-distance)))
    handles = [add_line(point_from_pose(start_pose), point_from_pose(end_pose), color=BLUE)]
    #handles.extend(draw_pose(start_pose))
    #handles.extend(draw_pose(end_pose))
    path = []
    pose_path = list(interpolate_poses(start_pose, end_pose, pos_step_size=0.01))
    for i, pose in enumerate(pose_path):
        print('Waypoint: {}/{}'.format(i+1, len(pose_path)))
        handles.extend(draw_pose(pose))
        conf = next(either_inverse_kinematics(robot, info, tool_link, pose, **kwargs), None)
        if conf is None:
            print('Failure!')
            path = None
            wait_for_user()
            break
        set_joint_positions(robot, ik_joints, conf)
        path.append(conf)
        wait_for_user()
        # for conf in islice(ikfast_inverse_kinematics(robot, info, tool_link, pose, max_attempts=INF, max_distance=0.5), 1):
        #    set_joint_positions(robot, joints[:len(conf)], conf)
        #    wait_for_user()
    remove_handles(handles)
    return path

def test_ik(robot, info, tool_link, tool_pose):
    draw_pose(tool_pose)
    # TODO: sort by one joint angle
    # TODO: prune based on proximity
    ik_joints = get_ik_joints(robot, info, tool_link)
    for conf in either_inverse_kinematics(robot, info, tool_link, tool_pose, use_pybullet=True,
                                          max_distance=INF, max_time=10, max_candidates=INF):
        # TODO: profile
        set_joint_positions(robot, ik_joints, conf)
        wait_for_user()

# plan motion
def plan(robot, info, tool_link, obstacles=[]):
    joints = get_ik_joints(robot, info, tool_link)
    start_pose = get_link_pose(robot, tool_link)
    # end_pose = multiply(start_pose, Pose(Point(z=-0.1)))
    end_pose = Pose([0.4, 0.0, 0.3], [np.pi, 0, np.pi])
    pose_path = list(interpolate_poses(start_pose, end_pose, pos_step_size=0.01))
    for i, pose in enumerate(pose_path):
        q_grasp = next(either_inverse_kinematics(robot, info, tool_link, pose, max_attempts=10000, max_time=5, obstacles=obstacles), None)
        if q_grasp is None:
            print('Failure!')
            path = None
            wait_for_user()
            break
        set_joint_positions(robot, joints, q_grasp)
        # wait_for_user()
    wait_for_user()

#####################################

def main():
    connect(use_gui=True)
    add_data_path()
    draw_pose(Pose(), length=1.)
    set_camera_pose(camera_point=[1, -1, 1])

    # define obstacles

    plane = p.loadURDF("plane.urdf")
    with LockRenderer():
        with HideOutput(True):
            robot = load_pybullet(FRANKA_URDF, fixed_base=True)
            assign_link_colors(robot, max_colors=3, s=0.5, v=1.)
            #set_all_color(robot, GREEN)
    
    obstacle_infos = [
        # box
        dict(
            size=[0.5, 0.5, 0.5],
            color=RGBA(0, 1, 0, 1),
            position=[0.5, 0.5, 0.5 / 2.],
            orientation=[0, 0, np.pi / 4]
        ),

        # table
        dict(
            size=[0.7, 0.7, 0.05],
            color=RGBA(202 / 255.0, 164 / 255.0, 114 / 255.0, 1),
            position=[0.5, 0.0, 0.1 - 0.05 / 2.0],
            orientation=[0, 0, 0],
        ),
    ]

    obstacles = []
    for obstacle_info in obstacle_infos:
        w, l, h = obstacle_info['size']
        obstacle = create_box(w=w, l=l, h=h, color=obstacle_info['color']) # Creates a red box obstacle
        set_point(obstacle, obstacle_info['position']) # Sets the [x,y,z] position of the obstacle
        set_euler(obstacle, obstacle_info['orientation']) #  Sets the [roll,pitch,yaw] orientation of the obstacle
        # print('Position:', get_point(obstacle))
        # print('Orientation:', get_euler(obstacle))

        obstacles.append(obstacle)
    # obstacles = [plane, obstacle] # TODO: collisions with the ground

    dump_body(robot)
    print('Start?')
    # wait_for_user()

    info = PANDA_INFO
    tool_link = link_from_name(robot, 'panda_hand')
    draw_pose(Pose(), parent=robot, parent_link=tool_link)
    joints = get_movable_joints(robot)
    print('Joints', [get_joint_name(robot, joint) for joint in joints])
    check_ik_solver(info)

    sample_fn = get_sample_fn(robot, joints)
    for i in range(10):
        print('Iteration:', i)
        conf = sample_fn()
        set_joint_positions(robot, joints, conf)
        # wait_for_user()
        #test_ik(robot, info, tool_link, get_link_pose(robot, tool_link))
        # test_retraction(robot, info, tool_link, use_pybullet=False,
        #                 max_distance=0.1, max_time=0.05, max_candidates=100)
        plan(robot, info, tool_link, obstacles)
    disconnect()

if __name__ == '__main__':
    main()
