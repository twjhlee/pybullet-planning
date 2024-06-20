from __future__ import print_function
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import sys

from pybullet_tools.franka_primitives import BodyPose, BodyConf, Command, get_grasp_gen, \
    get_ik_fn, get_free_motion_gen, get_holding_motion_gen
from pybullet_tools.utils import WorldSaver, enable_gravity, connect, dump_world, set_pose, \
    draw_global_system, draw_pose, set_camera_pose, Pose, Point, set_default_camera, stable_z, \
    BLOCK_URDF, load_model, wait_if_gui, disconnect, DRAKE_IIWA_URDF, wait_if_gui, update_state, disable_real_time, HideOutput, \
    load_pybullet, get_movable_joints, get_sample_fn, set_joint_positions

from pybullet_tools.ikfast.franka_panda.ik import FRANKA_URDF, PANDA_INFO

import numpy as np

def plan(robot, block, fixed, teleport):
    grasp_gen = get_grasp_gen(robot, 'top')
    ik_fn = get_ik_fn(robot, fixed=fixed, teleport=teleport, num_attempts=10000)
    free_motion_fn = get_free_motion_gen(robot, fixed=[block] + fixed, teleport=teleport)
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

IS_INIT = False
cur_joint_positions = None


def init_joint_state_cb(data):
    global IS_INIT, cur_joint_positions

    if IS_INIT:
        return

    # print(data)
    IS_INIT = True
    cur_joint_positions = data.position

def main(display='execute'): # control | execute | step
    rospy.init_node("panda_planner")
    rospy.Subscriber("/franka_ros_interface/custom_franka_state_controller/joint_states", JointState, init_joint_state_cb, queue_size=10)
    joint_position_pub = rospy.Publisher("/panda_joint_command", Float64MultiArray, queue_size=10)

    # wait until current robot pose is recieved.
    while not IS_INIT:
        rospy.sleep(0.1)
    rospy.loginfo("Current robot joint positions:\n\t%s", cur_joint_positions)

    connect(use_gui=True)
    disable_real_time()
    draw_global_system()
    with HideOutput():
        # robot = load_model(FRANKA_URDF) # KUKA_IIWA_URDF | DRAKE_IIWA_URDF
        robot = load_pybullet(FRANKA_URDF, fixed_base=True)
        floor = load_model('models/short_floor.urdf')
    set_pose(floor, Pose(Point(z=0.095)))
    joints = get_movable_joints(robot)

    objects = []

    # obstacle = create_box(w=0.05, l=0.05, h=0.05, color=RED) # Creates a red box obstacle
    # set_point(obstacle, []) # Sets the [x,y,z] position of the obstacle
    for i, x in enumerate(np.linspace(0.2, 0.5, 3)):
        for j, y in enumerate(np.linspace(-0.3, 0.3, 3)):
            if i == 1 and j == 1:
                target = load_model(BLOCK_URDF, fixed_base=False)
                set_pose(target, Pose(Point(x=x, y=y, z=stable_z(target, floor))))
            else:
                block = load_model(BLOCK_URDF, fixed_base=False)
                set_pose(block, Pose(Point(x=x, y=y, z=stable_z(block, floor))))
                objects.append(block)
                # set_pose(block, Pose(Point(x=x, y=y, z=stable_z(block, floor)), np.random.rand(3)*np.pi))
        
    set_default_camera(distance=2)
    dump_world()

    saved_world = WorldSaver()
    conf = cur_joint_positions + (0.01, 0.01) # last two positions indicate fingers
    set_joint_positions(robot, joints, conf)
    # command = plan(robot, block, fixed=[floor], teleport=False)
    command = plan(robot, target, fixed=objects, teleport=False)
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
        command.refine(num_steps=10).execute(time_step=0.005)
        # command.execute(time_step=0.05)
    elif display == 'step':
        command.step()
    elif display == 'publish':
        command.execute(time_step=1, joint_position_publisher=joint_position_pub)
        # command.execute(time_step=1, joint_position_publisher=joint_position_pub)
    else:
        raise ValueError(display)

    print('Quit?')
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main('publish')
