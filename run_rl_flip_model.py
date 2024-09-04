import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.high_level_commander import HighLevelCommander
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.mem import CompressedSegment, CompressedStart, MemoryElement
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.utils import uri_helper

# Initialize the drivers
cflib.crtp.init_drivers()

# Define the ActorCriticNet model class
class ActorCriticNet(nn.Module):
    def __init__(self):
        super(ActorCriticNet, self).__init__()
        
        self.actor_mlp = nn.Sequential(
            nn.Linear(23, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
        )
        self.mu = nn.Linear(128, 4)  # Assuming 4 action dimensions
        self.sigma = nn.Parameter(torch.zeros(4))  # Assuming 4 action dimensions
        
        # Value head
        self.value = nn.Linear(128, 1)
        
        # Normalization layers
        self.value_mean_std = nn.BatchNorm1d(1, affine=False)
        self.running_mean_std = nn.BatchNorm1d(23, affine=False)

    def forward(self, x):
        x = self.running_mean_std(x)  # Normalize input
        actor_features = self.actor_mlp(x)
        mu = self.mu(actor_features)
        sigma = torch.exp(self.sigma)
        value = self.value(actor_features)
        value = self.value_mean_std(value)  # Normalize value output
        return mu, sigma, value

# Instantiate the model
model = ActorCriticNet()

# Load the checkpoint
checkpoint = torch.load('Crazyflie.pth')
checkpoint_model = {k.replace('a2c_network.', ''): v for k, v in checkpoint['model'].items()}
unexpected_keys = ['value_mean_std.count', 'running_mean_std.count']
for key in unexpected_keys:
    if key in checkpoint_model:
        del checkpoint_model[key]
model.load_state_dict(checkpoint_model)
model.eval()

# URI to the Crazyflie to connect to
uri = uri_helper.uri_from_env(default='radio://0/80/2M')

def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

def quat_axis(q, axis=0):
    """
    Extract a specific axis from a quaternion rotation.

    Parameters:
    q (tensor of shape (4,) or (1, 4)): The quaternion in the format (qW, qX, qY, qZ).
    axis (int): The axis to extract (0 for X, 1 for Y, 2 for Z).

    Returns:
    tensor of shape (3,): The rotated basis vector representing the specified axis.
    """
    # Ensure the quaternion has shape (1, 4)
    if q.ndim == 1:  # If input is (4,)
        q = q.unsqueeze(0)  # Make it (1, 4)
    
    # Create a basis vector aligned with the specified axis, shape (1, 3)
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)  # (1, 3)
    basis_vec[:, axis] = 1  # Set the appropriate axis to 1

    # Rotate the basis vector using the quaternion, result will have shape (1, 3)
    rotated_vector = quat_rotate(q, basis_vec)

    # Remove the first dimension and return the result as (3,)
    return rotated_vector.squeeze(0)

def thrust_to_pwm(thrust_tensor):
    """
    Convert thrust values (in Newtons) given as a PyTorch tensor to PWM values (0 to 65535) for a Crazyflie motor.

    Parameters:
    thrust_tensor (torch.Tensor): A tensor containing thrust values in Newtons to convert to PWM.

    Returns:
    torch.Tensor: A tensor of corresponding PWM values.
    """
    # Coefficients from the thrust-to-PWM quadratic formula
    a = 2.130295e-11
    b = 1.032633e-6
    c = 5.484560e-4 - thrust_tensor

    # Calculate the discriminant
    discriminant = b**2 - 4 * a * c

    # Ensure the discriminant is non-negative; clamp to avoid invalid values
    discriminant = torch.clamp(discriminant, min=0)

    # Calculate the two solutions for PWM using the quadratic formula
    pwm1 = (-b + torch.sqrt(discriminant)) / (2 * a)
    pwm2 = (-b - torch.sqrt(discriminant)) / (2 * a)

    # Since PWM cannot be negative, choose the positive root
    pwm = torch.maximum(pwm1, pwm2)

    # Clamp PWM values to the valid range for Crazyflie (0 to 65535)
    pwm = torch.clamp(pwm, min=0, max=65535).to(torch.int)

    return pwm

def pre_physics_step(actions, root_rot, thrust_cmds_damp, thrust_rot_damp):
    EPS = 1e-6
    dt = 0.01

    motor_assymetry = np.array([1.0, 1.0, 1.0, 1.0])
    # Re-normalizing to sum-up to 4
    motor_assymetry = motor_assymetry * 4.0 / np.sum(motor_assymetry)

    thrust_max_coef = 9.81 * 0.033 * 7.5 * motor_assymetry / 4.0
    thrust_max = torch.tensor(thrust_max_coef, dtype=torch.float32)

    # Parameters for the controller
    motor_damp_time_up = 0.15
    motor_damp_time_down = 0.15

    # I use the multiplier 4, since 4*T ~ time for a step response to finish, where
    # T is a time constant of the first-order filter
    motor_tau_up = 4 * dt / (motor_damp_time_up + EPS)
    motor_tau_down = 4 * dt / (motor_damp_time_down + EPS)

    # Clamp to [-1.0, 1.0]
    thrust_cmds = torch.clamp(actions, min=-1.0, max=1.0)
    # Scale to [0.0, 1.0]
    thrust_cmds = (thrust_cmds + 1.0) / 2.0

    # Filtering the thruster and adding noise
    motor_tau = motor_tau_up * torch.ones(4, dtype=torch.float32)
    motor_tau[thrust_cmds < thrust_cmds_damp] = motor_tau_down
    motor_tau[motor_tau > 1.0] = 1.0

    # Since NN commands thrusts we need to convert to rotational velocity and back
    thrust_rot = thrust_cmds**0.5
    thrust_rot_damp = motor_tau * (thrust_rot - thrust_rot_damp) + thrust_rot_damp
    thrust_cmds_damp = thrust_rot_damp**2

    # Adding noise
    thrust_noise = 0.01 * torch.randn(4, dtype=torch.float32)
    thrust_noise = thrust_cmds * thrust_noise
    thrust_cmds_damp = torch.clamp(thrust_cmds_damp + thrust_noise, min=0.0, max=1.0)

    # Determine max thrust based on conditions
    thrusts = thrust_max * thrust_cmds_damp  # Shape (4,)

    return thrusts, thrust_cmds_damp, thrust_rot_damp  # mod_forces is (4, 3)

def check_flip_completion(prev_state, quat, prev_flips_number):
    root_quats = quat
    # Update the position reached indicator
    successful_flip_threshold = 0.1  # Stricter threshold for flip completion
    is_upright = (torch.abs(root_quats[0] - 1) < successful_flip_threshold) | (torch.abs(root_quats[0] + 1) < successful_flip_threshold)
    is_upside_down = (torch.abs(root_quats[2] - 1) < successful_flip_threshold) | (torch.abs(root_quats[2] + 1) < successful_flip_threshold)
    is_upright = is_upright.float()
    is_upside_down = is_upside_down.float()

    # Count successful flips
    flip_state=(prev_state == 2)
    flip_completion=torch.where(flip_state, is_upside_down, torch.zeros_like(is_upside_down))
    end_of_flip=torch.where(flip_state, is_upright, torch.zeros_like(is_upright))
    
    num_flips = prev_flips_number

    # Check if the number of flips reached 2
    double_flip = (num_flips >= 2)
    new_flips_number=torch.where(double_flip, prev_flips_number, prev_flips_number+flip_completion)

    end_double_flip = (end_of_flip==1)
    end_indicator= end_double_flip & double_flip
    double_flip = (new_flips_number>= 2)

    # Reset successful flip count if hover phase is reached
    new_flips_number = torch.where(
        double_flip,
        2*torch.ones(1, dtype=torch.float32),
        new_flips_number
    )
    return double_flip, new_flips_number

def check_up_position(prev_state, quat, prev_up_count):
    root_quats = quat  
    up_threshold = 0.1  

    # Extract the Z-axis direction vectors from the quaternions
    is_upright = (torch.abs(root_quats[0] - 1) < up_threshold) | (torch.abs(root_quats[0] + 1) < up_threshold)
    is_upright = is_upright.float()

    approach_state=(prev_state == 1)
    is_upright_completion=torch.where(approach_state, is_upright, torch.zeros_like(is_upright))
    uprigth= (is_upright_completion==1)

    # Update the up_count for each environment based on whether the drone is upright
    new_up_count = torch.where(
        uprigth,  
        torch.ones(1, dtype=torch.float32),  
        prev_up_count
    )

    return new_up_count

def get_states(position, quat, prev_state, prev_flips_number, prev_up_count):
    root_positions = position
    target_positions = torch.zeros(3, dtype=torch.float32)
    target_positions[2] = 1.5
    target_dist = torch.norm(target_positions - root_positions, dim=-1)
    left_height = 1.5-root_positions[2]
    
    current_up_count = check_up_position(prev_state, quat, prev_up_count)
    up_check = (current_up_count==1)
    approaching_target_mask = (left_height < 0.1) & (prev_state == 1) & up_check
    if approaching_target_mask.any():
        prev_state += 1

    flip_completed, current_flips_number = check_flip_completion(prev_state, quat, prev_flips_number)
    flipping_mask = (prev_state== 2) & flip_completed
    if flipping_mask.any():
        prev_state += 1

    return prev_state, current_flips_number, current_up_count

def set_motor_pwm(cf, m1_pwm, m2_pwm, m3_pwm, m4_pwm):
    """
    Sets the PWM values for each of the four motors and confirms they are set.
    """
    # Set motor parameters for Crazyflie using cf.param.set_value
    cf.param.set_value('motorPowerSet.m1', str(m1_pwm))
    cf.param.set_value('motorPowerSet.m2', str(m2_pwm))
    cf.param.set_value('motorPowerSet.m3', str(m3_pwm))
    cf.param.set_value('motorPowerSet.m4', str(m4_pwm))
    
    # Add a small delay to ensure parameters are set
    time.sleep(0.01)

    # Read back the parameters to verify
    m1 = cf.param.get_value('motorPowerSet.m1')
    m2 = cf.param.get_value('motorPowerSet.m2')
    m3 = cf.param.get_value('motorPowerSet.m3')
    m4 = cf.param.get_value('motorPowerSet.m4')
    
    print(f"PWM set and acknowledged: M1={m1}, M2={m2}, M3={m3}, M4={m4}")

def flush_command_queue(scf):
    """
    Flush any previous commands to prevent the Crazyflie from executing old commands.
    """
    print("Flushing command queue...")
    commander = scf.cf.commander
    commander.send_stop_setpoint()  # Stop any current commands
    time.sleep(0.1)  # Short delay to ensure stop command is registered


def fly_with_model(scf):
    """
    Function to control Crazyflie using the trained model.
    """
    # Initialize the commander for real-time thrust control
    commander = scf.cf.commander

    # Initialize state tensors for the first timestep
    previous_output = torch.zeros(4, dtype=torch.float32)  # Previous output
    prev_state = torch.tensor([1], dtype=torch.float32)  # Initial previous state
    prev_flips_number = torch.tensor([0], dtype=torch.float32)  # Initial flips number
    prev_up_count = torch.tensor([0], dtype=torch.float32)  # Initial up count
    thrust_cmds_damp = torch.zeros(4, dtype=torch.float32)
    thrust_rot_damp = torch.zeros(4, dtype=torch.float32)

    # Logging configurations to get sensor data (split into multiple configs)
    log_config1 = LogConfig(name='Stabilizer1', period_in_ms=10)
    log_config1.add_variable('stateEstimateZ.z', 'float')  # Height
    log_config1.add_variable('stateEstimateZ.x', 'float')  # Estimate in X
    log_config1.add_variable('stateEstimateZ.y', 'float')  # Estimate in Y

    log_config2 = LogConfig(name='Stabilizer2', period_in_ms=10)
    log_config2.add_variable('stateEstimate.qx', 'float')  # Quaternion X
    log_config2.add_variable('stateEstimate.qy', 'float')  # Quaternion Y
    log_config2.add_variable('stateEstimate.qz', 'float')  # Quaternion Z
    log_config2.add_variable('stateEstimate.qw', 'float')  # Quaternion W

    log_config3 = LogConfig(name='Stabilizer3', period_in_ms=10)
    log_config3.add_variable('stateEstimateZ.vx', 'float')  # Linear velocity X
    log_config3.add_variable('stateEstimateZ.vy', 'float')  # Linear velocity Y
    log_config3.add_variable('stateEstimateZ.vz', 'float')  # Linear velocity Z

    log_config4 = LogConfig(name='Stabilizer4', period_in_ms=10)
    log_config4.add_variable('stateEstimateZ.rateRoll', 'float')  # Angular velocity Roll
    log_config4.add_variable('stateEstimateZ.ratePitch', 'float')  # Angular velocity Pitch
    log_config4.add_variable('stateEstimateZ.rateYaw', 'float')  # Angular velocity Yaw

    # Using multiple loggers to retrieve data
    with SyncLogger(scf, log_config1) as logger1, SyncLogger(scf, log_config2) as logger2, \
         SyncLogger(scf, log_config3) as logger3, SyncLogger(scf, log_config4) as logger4:

        for log_entry1, log_entry2, log_entry3, log_entry4 in zip(logger1, logger2, logger3, logger4):
            # Retrieve sensor data from multiple log entries
            data1 = log_entry1[1]
            data2 = log_entry2[1]
            data3 = log_entry3[1]
            data4 = log_entry4[1]
            
            # Extract sensor data and convert to PyTorch tensors
            height = torch.tensor([data1['stateEstimateZ.z']], dtype=torch.float32)
            current_position = torch.tensor([
                data1['stateEstimateZ.x'],
                data1['stateEstimateZ.y'],
                data1['stateEstimateZ.z']
            ], dtype=torch.float32)
            
            quat = torch.tensor([
                data2['stateEstimate.qx'],
                data2['stateEstimate.qy'],
                data2['stateEstimate.qz'],
                data2['stateEstimate.qw']
            ], dtype=torch.float32)
            
            lin_vel = torch.tensor([
                data3['stateEstimateZ.vx'],
                data3['stateEstimateZ.vy'],
                data3['stateEstimateZ.vz']
            ], dtype=torch.float32)
            
            ang_vel = torch.tensor([
                data4['stateEstimateZ.rateRoll'],
                data4['stateEstimateZ.ratePitch'],
                data4['stateEstimateZ.rateYaw']
            ], dtype=torch.float32)

            # Get the current state of the drone
            current_state, current_flips_number, current_up_count = get_states(current_position, quat, prev_state, prev_flips_number, prev_up_count)

            # Extract rotation matrix axes
            rot_x = quat_axis(quat, 0)
            rot_y = quat_axis(quat, 1)
            rot_z = quat_axis(quat, 2)

            # Combine all data for input to the model in the correct order
            sensor_input = torch.cat((1.5 - height, previous_output, rot_x, rot_y, rot_z, lin_vel, ang_vel, current_state, prev_state, current_state - prev_state), dim=0).unsqueeze(0)

            # Use the model to generate an action based on the sensor input
            with torch.no_grad():
                mu, sigma, _ = model(sensor_input)
                action = mu.squeeze(0)

            # Update the previous output and state with the current model's output and current state
            previous_output = action
            prev_state = current_state
            prev_flips_number = current_flips_number
            prev_up_count = current_up_count

            # Convert model output to thrust commands
            print("Action: ", action)
            thrusts_vectors, thrust_cmds_damp, thrust_rot_damp = pre_physics_step(action, quat.unsqueeze(0), thrust_cmds_damp, thrust_rot_damp)
            print("Thrust vectors: ", thrusts_vectors)

            # Convert thrust vectors to PWM values
            pwm_values = thrust_to_pwm(thrusts_vectors)
            m1_pwm, m2_pwm, m3_pwm, m4_pwm = pwm_values.tolist()

            # Set the PWM values for the motors using the parameter setting function
            set_motor_pwm(scf.cf, m1_pwm, m2_pwm, m3_pwm, m4_pwm)
            print("PWM values set: ", m1_pwm, m2_pwm, m3_pwm, m4_pwm)

            # Sleep to maintain control loop timing
            time.sleep(0.01)

            # Stop after a certain duration for safety
            if time.time() - start_time > 10:
                break

    # Safely stop the Crazyflie after the loop ends
    commander.send_stop_setpoint()
    print("Commander stopped. Flight sequence complete.")

# Connect to Crazyflie and execute flight
with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
    flush_command_queue(scf)  # Flush any previous commands
    start_time = time.time()
    fly_with_model(scf)  # Execute the flight sequence