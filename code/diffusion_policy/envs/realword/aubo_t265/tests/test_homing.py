"""Offline home acceptance; no network or camera access."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from diffusion_policy.envs.realword.aubo_homing import JointHome, joints, load_home
from diffusion_policy.envs.realword.aubo_interpolation_controller import AuboInterpolationController, Command


class HomeTests(unittest.TestCase):
    def test_recorded_pair_matches(self):
        config=json.loads((Path(__file__).resolve().parents[1] / 'config/dual.json').read_text())
        for arm in config['arms']:
            q=load_home(Path(__file__).resolve().parents[1] / 'config/home_positions.json',arm['ip'],arm['serial'])
            self.assertEqual(q.shape,(6,))
        with self.assertRaises(ValueError):
            load_home(Path(__file__).resolve().parents[1] / 'config/home_positions.json','192.168.2.200','wrong')

    def test_invalid_joints(self):
        for value in ([0]*5,[float('nan')]*6,[float('inf')]*6):
            with self.assertRaises(ValueError): joints(value)

    def test_trajectory_speed_acceleration_endpoints(self):
        plan=JointHome(np.zeros(6),[.8,-.6,.4,-.3,.2,0])
        samples=[plan.start]
        for _ in range(int(np.ceil(plan.duration/.02))+5):
            samples.append(plan.sample(.02)); plan.accepted(.02)
        samples=np.array(samples)
        self.assertLessEqual(np.max(np.abs(np.diff(samples,axis=0)/.02)),.15+1e-9)
        self.assertLessEqual(np.max(np.abs(np.diff(samples,n=2,axis=0)/.02**2)),.3+1e-6)
        np.testing.assert_allclose(samples[-1],plan.target)

    def test_no_modulo_joint_wrap(self):
        plan=JointHome(np.ones(6)*3.5,np.ones(6)*-3.)
        np.testing.assert_allclose(plan.target,-3.)
        self.assertGreater(plan.duration,6.5/.15)

    def test_queue_rejection_does_not_advance(self):
        plan=JointHome(np.zeros(6),np.ones(6))
        a=plan.sample(.02)
        np.testing.assert_array_equal(a,plan.sample(.02))
        self.assertEqual(plan.elapsed,0)
        plan.accepted(.02)
        self.assertTrue(np.all(plan.sample(.02)>a))

    def test_requires_five_fresh_settled_samples(self):
        plan=JointHome(np.zeros(6),np.ones(6)*.01)
        plan.accepted(plan.duration)
        for i in range(4):
            self.assertFalse(plan.check(plan.target,np.zeros(6),i,plan.target))
        for _ in range(8):
            self.assertFalse(plan.check(plan.target,np.zeros(6),3,plan.target))
        self.assertTrue(plan.check(plan.target,np.zeros(6),4,plan.target))

    def test_not_done_when_moving(self):
        plan=JointHome(np.zeros(6),np.ones(6)*.01)
        plan.accepted(plan.duration)
        for i in range(10):
            self.assertFalse(plan.check(plan.target,np.ones(6)*.1,i,plan.target))

    def test_following_error_and_timeout(self):
        plan=JointHome(np.zeros(6),np.ones(6))
        with self.assertRaisesRegex(RuntimeError,'following error'):
            plan.check(np.ones(6),np.zeros(6),0,np.zeros(6))
        plan.deadline=0
        with self.assertRaisesRegex(RuntimeError,'timeout'):
            plan.check(np.zeros(6),np.zeros(6),0,np.zeros(6))

    def test_zero_distance(self):
        plan=JointHome(np.ones(6),np.ones(6))
        np.testing.assert_allclose(plan.sample(.02),np.ones(6))


class ControllerTests(unittest.TestCase):
    def rig(self, resume=False):
        motion=Mock()
        motion.startMove.return_value=0
        algorithm=SimpleNamespace(inverseKinematics=Mock(return_value=(np.ones(6)*.102,0)))
        robot=SimpleNamespace(getMotionControl=lambda:motion,getRobotAlgorithm=lambda:algorithm)
        rpc=Mock()
        rpc.hasConnected.return_value=rpc.hasLogined.return_value=True
        rpc.getRobotNames.return_value=['R1']
        rpc.getRobotInterface.return_value=robot
        sdk=SimpleNamespace(RpcClient=lambda:rpc,returnValue2Str=lambda code:str(code))
        last=[np.ones(6)*.1]
        calls=[]
        received=[time.time()]
        def state(*args):
            received[0] += .00001
            return dict(eef_pose=np.array([.5,0,0,0,0,0]),eef_speed=np.zeros(6),
                        joint_angles=last[0].copy(),joint_speeds=np.zeros(6),received=received[0])
        def send(_,q):
            calls.append(q.copy())
            if len(calls)==1: return 2
            last[0]=q.copy()
            return 0
        def command(cmd,pose=None):
            return dict(cmd=np.array([cmd]),target_pose=np.array([np.zeros(6) if pose is None else pose]),
                        duration=np.array([.02]),target_time=np.array([0.]))
        commands=[command(Command.HOME.value,np.ones(6)*.102)]
        # Old queued Cartesian target must not interrupt home.
        commands.append(command(Command.SERVOL.value,np.array([.8,0,0,0,0,0])))
        commands.extend(command(Command.HEARTBEAT.value) for _ in range(65))
        if resume:
            commands.append(command(Command.SERVOL.value,np.array([.5,0,0,0,0,0])))
        commands.append(command(Command.STOP.value))
        rig=SimpleNamespace(soft_real_time=False,robot_ip='offline',robot_port=30004,
            stop_event=SimpleNamespace(is_set=lambda:False),gripper_config={'type':'disabled'},
            max_joint_speed=.5,max_following_error=.2,reject_joint_jumps=False,
            username='test',password='test',speed_fraction=.2,_power_on=Mock(),
            _start_rtde=Mock(return_value=(Mock(),0,None,None)),_copy_rtde_state=state,
            _start_runtime=Mock(),_warmup_servo=Mock(),frequency=50.,ring_buffer=Mock(),
            startup_ok=SimpleNamespace(value=False),ready_event=Mock(),home_state=SimpleNamespace(value=0),
            input_queue=SimpleNamespace(get_all=Mock(side_effect=commands)),command_timeout=.5,
            max_joint_step=.05,max_pos_speed=.05,max_rot_speed=.3,_servo_joint=send,
            _set_servo_mode=Mock(),_limit_joint_step=AuboInterpolationController._limit_joint_step,verbose=False)
        return rig,sdk,algorithm,calls

    def test_home_uses_joints_ignores_cartesian_and_holds(self):
        rig,sdk,algo,calls=self.rig()
        with patch('diffusion_policy.envs.realword.aubo_interpolation_controller.require_pyaubo_sdk',return_value=sdk), patch('diffusion_policy.envs.realword.aubo_interpolation_controller.time.sleep'):
            AuboInterpolationController.run(rig)
        algo.inverseKinematics.assert_not_called()
        np.testing.assert_array_equal(calls[0],calls[1])
        np.testing.assert_allclose(calls[-1],.102)
        self.assertEqual(rig.home_state.value,2)
        self.assertFalse(rig._set_servo_mode.call_args.args[1])

    def test_new_reanchored_command_releases_home_hold(self):
        rig,sdk,algo,calls=self.rig(resume=True)
        with patch('diffusion_policy.envs.realword.aubo_interpolation_controller.require_pyaubo_sdk',return_value=sdk), patch('diffusion_policy.envs.realword.aubo_interpolation_controller.time.sleep'):
            AuboInterpolationController.run(rig)
        algo.inverseKinematics.assert_called_once()
        self.assertEqual(rig.home_state.value,0)

    def test_watchdog_stops_home(self):
        rig,sdk,algo,calls=self.rig()
        rig.command_timeout=-1
        with patch('diffusion_policy.envs.realword.aubo_interpolation_controller.require_pyaubo_sdk',return_value=sdk):
            with self.assertRaisesRegex(RuntimeError,'watchdog'):
                AuboInterpolationController.run(rig)
        self.assertEqual(calls,[])
        self.assertFalse(rig._set_servo_mode.call_args.args[1])

    def test_stop_preempts_home(self):
        rig,sdk,algo,calls=self.rig()
        rig.input_queue.get_all.side_effect=[dict(cmd=np.array([Command.HOME.value,Command.STOP.value]))]
        with patch('diffusion_policy.envs.realword.aubo_interpolation_controller.require_pyaubo_sdk',return_value=sdk):
            AuboInterpolationController.run(rig)
        self.assertEqual(calls,[])

    def test_hold_discards_queued_targets_and_keeps_joint_command(self):
        rig,sdk,algo,calls=self.rig()
        rig.input_queue.get_all.side_effect=[dict(cmd=np.array([Command.SERVOL.value, Command.HOLD.value])),
            dict(cmd=np.array([Command.HEARTBEAT.value])),dict(cmd=np.array([Command.STOP.value]))]
        with patch('diffusion_policy.envs.realword.aubo_interpolation_controller.require_pyaubo_sdk',return_value=sdk), patch('diffusion_policy.envs.realword.aubo_interpolation_controller.time.sleep'):
            AuboInterpolationController.run(rig)
        algo.inverseKinematics.assert_not_called()
        for call in calls: np.testing.assert_allclose(call, .1)

    def test_rtde_staleness_stops_both_home_and_cartesian(self):
        rig,sdk,algo,calls=self.rig()
        original=rig._copy_rtde_state
        reads=[0]
        def stale(*args):
            state=original(*args)
            reads[0] += 1
            if reads[0]>1: state['received']=time.time()-1
            return state
        rig._copy_rtde_state=stale
        with patch('diffusion_policy.envs.realword.aubo_interpolation_controller.require_pyaubo_sdk',return_value=sdk):
            with self.assertRaisesRegex(RuntimeError,'stale'):
                AuboInterpolationController.run(rig)
        self.assertEqual(calls,[])


if __name__=='__main__':
    unittest.main()
