import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# os.environ['JAX_PLATFORMS'] = 'cpu' 
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
from jax import numpy as jnp
from jax import random
from difai.aif import AIF_Agent, AIF_Simulation
from iaif.mouse_simple import Mouse_Cursor
import numpy as np
import pickle
import time
import pandas as pd
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

# Load targets
targets = pd.read_csv("./data/targets.csv", header=0).values[:, :2]
start_target = np.array([0.   , 0.003])

run_name = "VFE_EFE_sweep_5_restarts"

out_folder = f"./data/simulations/{run_name}"

TARGETS = [0,1,2,3,4,5,6,7,8,9,10,11]
DIV_THRESHOLDS = [None] # [None, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
EFE_THRESHOLDS = [None, 1.0]
VFE_THRESHOLDS = [None, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16]
NUMBER_PLANS = [5000] 
MINIMAL_OPEN_LOOP_STEPS = 0
REACTION_TIME = 0.1
NUM_REPEATS = 5 # 10
NUMSTEPS = 100

target_id = TARGETS[0]

# Create Generative Process (real system)
buttons = [start_target,targets[target_id]]

# Set x0 
x0 = jnp.array([buttons[0][0], 0.0, buttons[1][0], buttons[1][1]])
k = np.float64(0.0)
d = np.float64(24)
sys_params = jnp.array([k, d])
dt = 0.02
mouse_cursor = Mouse_Cursor(x0, *sys_params, dt=dt)
mouse_cursor.reset()

# Create Generative Model (Mouse Cursor Model)
mouse_cursor_model = Mouse_Cursor(x0, *sys_params, dt=dt)
mouse_cursor_model.reset()

noise_params = {}
noise_params['observation_std'] = {'id': np.array([0,1,2,3])} # standard deviation of the applied observation noise

# Create AIF Agent with very good initial belief
agent = AIF_Agent(generative_model=mouse_cursor_model, noise_params=noise_params)

# Reset beliefs
initial_belief_sys_cov = jnp.diag(
    (jnp.array([0.000001, 2.0]))**2
)
initial_belief_state_cov = jnp.diag(
    (jnp.array([0.05, 0.001, 0.45, 0.02]))**2
)


agent.set_params_with_defaults(n_steps_o=30, 
                 lr_o=[3e-4,1e-2,3e-4], # Different learning rates for cursor and target states
                 n_samples_o=300,
                 use_complete_ukf=True, 
                 use_ukf_obs_pref=True,
                  n_samples_obs_pref_o=2,
                  n_samples_obs_pref_s=50,
                  select_max_pi=True,
                  horizon=12,
                  grouped_state_updates=[(0,1),(2,),(3,)], # Update cursor and target states separately
                  exlude_observation_indices = [[],np.array([1]),np.array([1])],
                  ic_use_remaining_nefe=True,
                  action_prior= [jnp.array([0.0]), jnp.array([50**2])], # set normal distribution as prior for action selection
                reaction_time=REACTION_TIME,
                ic_minimal_open_loop_steps=MINIMAL_OPEN_LOOP_STEPS,
                n_samples_C = 200,
                ic_efe_type='fixed'
                )

agent.set_initial_beliefs(initial_belief_state=[x0.at[2:].set(start_target),initial_belief_state_cov], 
                          initial_belief_noise=[jnp.log(jnp.array([0.001, 0.000001, 0.000001, 0.000001])), 0.00001*jnp.eye(agent.params['dim_noise'])],                          
                          initial_belief_sys=[sys_params, initial_belief_sys_cov])

# Preference distribution for observing button click and not observing missclick
agent.set_preference_distribution(C=[jnp.array([1.0, 1.0]), jnp.diag(jnp.array([0.001**2, 0.0001**2]))], 
                                   C_index=[0,1],
                                   sys_dependent_C=None,
                                   state_dependent_C=np.array([[0],[2]]),
                                   use_observation_preference=True)
agent.params['ii_threshold'] = VFE_THRESHOLDS
agent.initialize()

print("**** Starting simulations ****")

# Create AIF Simulation
for num_plans in tqdm(NUMBER_PLANS, "Plan Number", leave=True):
    agent.params['n_plans'] = num_plans
    for div_threshold in tqdm(DIV_THRESHOLDS, "Div Threshold", leave=False):
        agent.params['ic_div_threshold'] = div_threshold
        for efe_threshold in tqdm(EFE_THRESHOLDS, "EFE Threshold", leave=False):
            agent.params['ic_efe_threshold'] = efe_threshold
            key = random.PRNGKey(42)
            for vfe_threshold in tqdm(VFE_THRESHOLDS, "VFE Threshold", leave=False):
                agent.params['ii_threshold'] = vfe_threshold
                key = random.PRNGKey(42)
                for target_id in tqdm(TARGETS, "Targets", leave=False):
                    # Create Generative Process (real system)
                    buttons = [start_target,targets[target_id]]
                    # Set x0 to the other button
                    x0 = jnp.array([buttons[0][0], 0.0, buttons[1][0], buttons[1][1]])
                    sys_params = jnp.array([k, d])

                    mouse_cursor = Mouse_Cursor(x0, *sys_params, dt=dt)

                    # Create Markov Blanket between real system and agent
                    noise_params = {}
                    noise_params['observation_std'] = {'id': np.array([0]), 'value': jnp.array([0.001])}# (0, 0.05) # standard deviation of the applied observation noise

                    sim = AIF_Simulation(agent, mouse_cursor, noise_params)
                    for repeat in tqdm(range(NUM_REPEATS), "repeat", leave=False):

                        save_path = f"{out_folder}/target_{target_id}_nplans_{num_plans}_pred_{div_threshold}_prag_{efe_threshold}_inf_{vfe_threshold}_rep_{repeat}.pkl"
                        if os.path.exists(save_path):
                            print(f"File {save_path} already exists. Skipping...")
                            continue

                        use_key, key = random.split(key)
                        t0 = time.time()
                        (bb, bb_after_rt, xx, oo, aa, aa_applied, lll, NEFE_PLAN, 
                        PRAGMATIC_PLAN, INFO_GAIN_PLAN, NEFES, PRAGMATICS, INFO_GAINS, 
                        ic_timesteps, ic_pred_error, IC_CRITERIA, bb_predicted, CUR_PRAGMATICS, 
                        CUR_PLAN, II_F, II_FIRED) = sim.run_iaif(numsteps=NUMSTEPS, verbose=False, key=use_key)        
                        t1 = time.time()
                        create_dir = os.path.dirname(save_path)
                        if not os.path.exists(create_dir):
                            os.makedirs(create_dir)
                            print(f"Directory {create_dir} created.")
                            
                        with open(save_path, 'wb') as f:
                            pickle.dump({
                                'xx': xx,
                                'oo': oo,
                                'bb': bb,
                                'bb_after_rt': bb_after_rt,
                                'aa': aa,
                                'aa_applied': aa_applied,
                                'lll': lll,
                                'nefe_plan': NEFE_PLAN,
                                'pragmatic_plan': PRAGMATIC_PLAN,
                                'info_gain_plan': INFO_GAIN_PLAN,
                                'belief_noise': agent.belief_noise,
                                'params': agent.params,
                                'sys_params_real': sys_params,
                                'noise_params_real': noise_params,
                                'noise_params_model': noise_params,
                                'buttons': buttons,
                                'dt': dt,
                                'ic_div_threshold': agent.params['ic_div_threshold'],
                                'ic_efe_threshold': agent.params['ic_efe_threshold'],
                                'reaction_time': agent.params['reaction_time'],
                                'ic_timesteps': ic_timesteps,
                                'ic_pred_error': ic_pred_error,
                                'ic_criteria': IC_CRITERIA,
                                'bb_predicted': bb_predicted,
                                'computation_time': t1 - t0,
                                'cur_pragmatics': CUR_PRAGMATICS,
                                'cur_plan': CUR_PLAN,
                                'ii_threshold': agent.params['ii_threshold'],
                                'ii_F': II_F,
                                'ii_fired': II_FIRED
                            }, f)
                        print(f"Simulation for target {target_id}, DIV threshold {div_threshold}, EFE threshold {efe_threshold}, VFE threshold {vfe_threshold}, repeat {repeat} completed in {t1 - t0:.2f} seconds. Results saved to {save_path}.")
                    print(f"--- Completed simulations for target {target_id} ---")
                print(f"--- Completed simulations for VFE {vfe_threshold} ---")
            print(f"=== Completed simulations for DIV threshold {div_threshold}, EFE threshold {efe_threshold} ===")
print("**** All simulations completed. ****")