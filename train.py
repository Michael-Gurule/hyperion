import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import os
import torch
import numpy as np
from collections import defaultdict
import logging
from tqdm import tqdm

# Import Environment and Configs
from src.env.scaled_environment import (
    ScaledHypersonicSwarmEnv,
    AdversarialConfig,
    RewardConfig,
)
from src.models.curriculum import (
    ParameterizedCurriculumScheduler,
    ParameterizedCurriculumConfig,
)
from src.models.intrinsic_rewards import (
    IntrinsicRewardConfig,
    IntrinsicRewardCalculator,
)

# Import Sensor Fusion Model
from src.models.detection import SensorFusion

# Import GNN Communication
from src.models.gnn_communication import SwarmGNN, CoordinationHead

# Import MAPPO
from src.models.mappo import MAPPO, MAPPOConfig

# Import Hierarchical MAPPO
from src.models.hierarchical_policy import (
    HierarchicalPolicyConfig,
    HierarchicalMAPPO,
    RoleType,
)

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HYPERION")


def create_env_from_hydra(cfg: DictConfig, curriculum=None) -> ScaledHypersonicSwarmEnv:
    """Helper to build the environment using Hydra config."""

    # 1. Resolve Curriculum Settings
    if curriculum:
        env_config = curriculum.get_env_config()
        adv_config = curriculum.get_adversarial_config()
    else:
        env_config = {
            "target_speed": cfg.environment.target.speed,
            "evasion_probability": 0.0,
            "target_behavior": "BALLISTIC",
        }
        adv_config = {"enabled": False}

    # 2. Build Adversarial Config
    adv_enabled = cfg.environment.get("adversarial", {}).get("enabled", False)

    adversarial = AdversarialConfig(
        enabled=adv_config.get("enabled", adv_enabled),
        evasion_probability=adv_config.get("evasion_probability", 0.0),
        evasive_maneuvers=adv_config.get("evasive_maneuvers", False),
        jink_frequency=adv_config.get("jink_frequency", 0.0),
        jink_magnitude=adv_config.get("jink_magnitude", 0.0),
    )

    # 3. Build Reward Config
    rewards = RewardConfig(
        intercept_reward=100.0,
        distance_scale=0.1,
        fuel_penalty=0.01,
        formation_bonus=0.5,
        projectile_hit_bonus=50.0,
        projectile_launch_cost=-0.5,
    )

    return ScaledHypersonicSwarmEnv(
        num_agents=cfg.environment.num_agents,
        max_steps=cfg.environment.max_steps,
        arena_size=cfg.environment.arena_size,
        target_speed=env_config["target_speed"],
        adversarial_config=adversarial,
        reward_config=rewards,
        use_projectiles=True,
    )


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(cfg: DictConfig):
    # --- 1. MLOPS SETUP ---
    # Convert config to native dict for W&B
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Initialize W&B
    run = wandb.init(
        project=cfg.project_name,
        name=cfg.experiment_name,
        config=config_dict,
        job_type="training",
        tags=[cfg.agent.algorithm, cfg.environment.name, cfg.hardware.name],
        monitor_gym=True,  # Auto-record videos if render is enabled
    )

    # Determine Device (MPS for Mac, Cuda for NVIDIA)
    device = torch.device(cfg.hardware.device)
    logger.info(f"🚀 Initializing HYPERION on {device}")

    # --- 2. COMPONENT INITIALIZATION ---

    # A. Curriculum
    curriculum = None
    if cfg.curriculum.enabled:
        logger.info("📚 Initializing Curriculum Scheduler...")
        # Map YAML stages to the Curriculum Config object

        curr_config = ParameterizedCurriculumConfig(
            agent_max_speed=cfg.environment.agent.max_speed,
            advancement_threshold=0.7,  # Defaulting for stability
            regression_threshold=0.3,
        )
        curriculum = ParameterizedCurriculumScheduler(config=curr_config)

    # B. Environment
    env = create_env_from_hydra(cfg, curriculum)

    # Get Dimensions
    sample_obs = env.reset()[0]
    sample_agent_id = list(sample_obs.keys())[0]
    obs_dim = sample_obs[sample_agent_id].shape[0]
    # 3 dims if projectiles (dx, dy, shoot), 2 if not.
    action_dim = 3

    # C. Intrinsic Rewards
    intrinsic_calculator = None
    # Defaut to True for Base line; in practice, use cfg.intrinsic.enabled
    if True:
        logger.info("🧠 Initializing Intrinsic Motivation...")
        int_cfg = IntrinsicRewardConfig(
            trailing_penalty_scale=0.5,
            geometry_bonus_scale=1.0,
            novelty_bonus_scale=0.1,
        )
        intrinsic_calculator = IntrinsicRewardCalculator(config=int_cfg)

    # E. Detection / Sensor Fusion
    sensor_fusion = None
    if cfg.detection.enabled:
        logger.info("🎯 Initializing Sensor Fusion...")
        sensor_fusion = SensorFusion(
            state_dim=cfg.detection.kalman.state_dim,
            process_noise=cfg.detection.kalman.process_noise,
            measurement_noise=cfg.detection.kalman.measurement_noise,
        )

    # F. GNN Communication Layer
    swarm_gnn = None
    coordination_head = None
    if cfg.communication.enabled:
        logger.info("📡 Initializing GNN Communication Layer...")
        swarm_gnn = SwarmGNN(
            obs_dim=obs_dim,
            hidden_dim=cfg.communication.gnn.hidden_dim,
            embed_dim=cfg.communication.gnn.embed_dim,
            num_gnn_layers=cfg.communication.gnn.num_layers,
            heads=cfg.communication.gnn.heads,
            dropout=cfg.communication.gnn.dropout,
            communication_range=cfg.communication.graph.communication_range,
        ).to(device)

        coordination_head = CoordinationHead(
            embed_dim=cfg.communication.gnn.embed_dim,
            hidden_dim=cfg.communication.gnn.hidden_dim,
            num_roles=cfg.communication.coordination.num_roles,
        ).to(device)

    # G. Agent / Policy (The "Brain")
    if cfg.agent.name == "mappo_gnn" or cfg.agent.get("use_hierarchical", False):
        logger.info("🤖 Loaded Hierarchical MAPPO Agent")
        policy_config = HierarchicalPolicyConfig(
            action_dim=action_dim,
            role_update_interval=10,
            manager_lr_scale=0.5,
        )
        trainer = HierarchicalMAPPO(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_agents=cfg.environment.num_agents,
            config=policy_config,
            device=device,
        )
    else:
        logger.info(f"🤖 Loaded Standard {cfg.agent.algorithm} Agent")
        mappo_config = MAPPOConfig(
            actor_lr=cfg.agent.hyperparameters.lr,
            critic_lr=cfg.agent.hyperparameters.lr * 3,
            gamma=cfg.agent.hyperparameters.gamma,
            gae_lambda=cfg.agent.hyperparameters.lambda_gae,
            clip_param=cfg.agent.hyperparameters.clip_param,
            entropy_coeff=cfg.agent.hyperparameters.entropy_coeff,
            num_epochs=cfg.agent.hyperparameters.num_sgd_iter,
            device=device,
        )
        trainer = MAPPO(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_agents=cfg.environment.num_agents,
            config=mappo_config,
        )

    # --- 3. TRAINING LOOP ---
    total_timesteps = 0
    episodes = 0

    # Create checkpoint directory based on Hydra output
    ckpt_dir = os.getcwd()  # Hydra changes CWD to 'outputs/date/time'
    logger.info(f"💾 Checkpoints will save to: {ckpt_dir}")

    try:
        pbar = tqdm(
            total=cfg.evaluation.episodes,
            desc="Training",
            unit="ep",
            dynamic_ncols=True,
        )
        while (
            episodes < cfg.evaluation.episodes
        ):  # Using evaluation count for short test, usually total_timesteps
            observations, _ = env.reset()
            if intrinsic_calculator:
                intrinsic_calculator.reset()
            if sensor_fusion:
                sensor_fusion.reset()

            episode_reward = 0
            episode_detection_confidence = 0.0
            episode_coordination_score = 0.0
            gnn_role_distribution = None
            steps = 0

            # --- Rollout ---
            for step in range(cfg.hardware.rollout_fragment_length):
                # --- NaN Protection: Sanitize observations before policy forward pass ---
                for agent_id, obs in list(observations.items()):
                    if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
                        logger.warning(
                            f"Step {step}: NaN/Inf in obs for {agent_id}, replacing with zeros"
                        )
                        observations[agent_id] = np.zeros_like(obs)

                # --- GNN Communication: Enhance observations with swarm context ---
                coordination_score = 0.0

                if swarm_gnn is not None:
                    # Extract agent positions from environment
                    agent_positions = []
                    agent_obs_list = []
                    agent_ids = list(observations.keys())

                    for agent_id in agent_ids:
                        obs = observations[agent_id]
                        agent_obs_list.append(obs)
                        # Positions are first 2 dims of observation (normalized)
                        # Denormalize: obs[:2] * arena_size
                        pos = obs[:2] * cfg.environment.arena_size
                        agent_positions.append(pos)

                    if len(agent_positions) > 0:
                        # Convert to tensors
                        obs_tensor = torch.tensor(
                            np.array(agent_obs_list), dtype=torch.float32, device=device
                        )
                        pos_tensor = torch.tensor(
                            np.array(agent_positions), dtype=torch.float32, device=device
                        )

                        # Forward through GNN
                        with torch.no_grad():
                            gnn_embeddings = swarm_gnn(obs_tensor, pos_tensor)

                            # Get coordination metrics
                            enhanced, role_probs, coord_score = coordination_head(
                                gnn_embeddings,
                                return_roles=cfg.communication.coordination.return_roles,
                            )

                            coordination_score = (
                                coord_score.item()
                                if coord_score.dim() == 0
                                else coord_score.mean().item()
                            )

                            if role_probs is not None:
                                # Track role distribution [scout, tracker, interceptor, support]
                                gnn_role_distribution = role_probs.mean(dim=0).cpu().numpy()

                episode_coordination_score += coordination_score

                # Action Selection
                if hasattr(trainer, "select_actions"):
                    # Handle difference between Hierarchical (returns 4 items) and Standard (returns 3)
                    result = trainer.select_actions(observations)
                    if len(result) == 4:
                        actions, log_probs, values, roles = result
                    else:
                        actions, log_probs, values = result
                        roles = {}

                # Environment Step
                next_obs, rewards, terminations, truncations, infos = env.step(actions)

                # --- Sensor Fusion: Filter target position estimates ---
                fused_target_pos = None
                fused_target_vel = None
                detection_confidence = 0.0

                if sensor_fusion and env.target_state is not None:
                    # Get raw target position from environment
                    if isinstance(env.target_state, dict):
                        raw_pos = env.target_state.get("position", np.zeros(2))
                    else:
                        raw_pos = env.target_state[:2]

                    # Add measurement noise to simulate sensor uncertainty
                    noise_std = cfg.detection.sensor.position_noise_std
                    noisy_measurement = raw_pos + np.random.normal(0, noise_std, size=2)

                    # Predict and update Kalman filter
                    dt = cfg.detection.sensor.dt
                    sensor_fusion.predict(dt)
                    sensor_fusion.update(noisy_measurement)

                    # Get filtered estimates
                    fused_target_pos = sensor_fusion.get_position()
                    fused_target_vel = sensor_fusion.get_velocity()

                    # Compute confidence based on filter covariance
                    position_uncertainty = np.sqrt(np.trace(sensor_fusion.P[:2, :2]))
                    detection_confidence = max(0.0, 1.0 - position_uncertainty / 1000.0)

                # --- NaN Protection: Sanitize next_obs to prevent buffer corruption ---
                for agent_id, obs in list(next_obs.items()):
                    if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
                        logger.warning(
                            f"Step {step}: NaN/Inf in next_obs for {agent_id}, replacing with zeros"
                        )
                        next_obs[agent_id] = np.zeros_like(obs)

                # Intrinsic Reward Calculation
                combined_rewards = rewards.copy()
                if intrinsic_calculator and env.target_state is not None:
                    # 1. Get Target Data (Handle dictionary or array format for robustness)
                    if isinstance(env.target_state, dict):
                        t_pos = env.target_state.get("position", np.zeros(2))
                        t_vel = env.target_state.get("velocity", np.zeros(2))
                    else:
                        # Fallback if target_state is just an array [x, y, vx, vy]
                        t_pos = env.target_state[:2]
                        t_vel = env.target_state[2:4]

                    # 2. Loop through all active agents to calculate individual intrinsic rewards
                    for agent_id, obs in observations.items():
                        # Skip if agent is not in the reward dict (dead/done)
                        if agent_id not in combined_rewards:
                            continue

                        # 3. Extract Agent State from Observation
                        # Assumption: Obs is standard [x, y, vx, vy, ...]
                        # We default to first 4 indices.
                        a_pos = obs[:2]
                        a_vel = obs[2:4]

                        # Skip intrinsic calculation if observation contains NaN/Inf
                        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
                            logger.warning(
                                f"Skipping intrinsic reward for {agent_id}: invalid obs"
                            )
                            continue

                        # 4. Compute the Intrinsic Reward
                        # FIXED: Use 'compute_agent_reward' and explicit argument names
                        intrinsic_reward, components = (
                            intrinsic_calculator.compute_agent_reward(
                                agent_id=agent_id,
                                agent_position=a_pos,
                                agent_velocity=a_vel,
                                target_position=t_pos,
                                target_velocity=t_vel,
                                detection_range=2000.0,
                                include_novelty=True,
                            )
                        )
                        combined_rewards[agent_id] += intrinsic_reward

                # --- NaN Protection: Sanitize rewards to prevent gradient corruption ---
                nan_count = 0
                for agent_id in list(combined_rewards.keys()):
                    reward_val = combined_rewards[agent_id]
                    if np.isnan(reward_val) or np.isinf(reward_val):
                        nan_count += 1
                        combined_rewards[agent_id] = 0.0
                    else:
                        # Clip extreme values to prevent exploding gradients
                        combined_rewards[agent_id] = float(
                            np.clip(reward_val, -100.0, 100.0)
                        )

                if nan_count > 0:
                    logger.warning(
                        f"Step {step}: Sanitized {nan_count} NaN/Inf rewards"
                    )

                dones = {
                    a: terminations.get(a, False) or truncations.get(a, False)
                    for a in observations
                }

                # Store Transition
                if hasattr(trainer, "store_transition"):
                    if roles:
                        trainer.store_transition(
                            observations,
                            actions,
                            combined_rewards,
                            dones,
                            log_probs,
                            values,
                            roles,
                        )
                    else:
                        trainer.store_transition(
                            observations,
                            actions,
                            combined_rewards,
                            dones,
                            log_probs,
                            values,
                        )

                episode_reward += np.mean(list(rewards.values()))
                episode_detection_confidence += detection_confidence
                episode_reward += np.mean(list(combined_rewards.values()))
                observations = next_obs
                steps += 1
                total_timesteps += len(observations)

                if all(dones.values()) or len(observations) == 0:
                    break

            # --- Update & Log ---
            if hasattr(trainer, "update"):
                # Bootstrap value for last observation
                _, _, last_values = (
                    trainer.select_actions(observations)
                    if not roles
                    else trainer.select_actions(observations)[:3]
                )

                # The config was passed during init, so we don't need to pass hyperparameters again
                update_stats = trainer.update(last_values)

                # Log to W&B
                avg_detection_confidence = (
                    episode_detection_confidence / steps if steps > 0 else 0.0
                )
                avg_coordination_score = (
                    episode_coordination_score / steps if steps > 0 else 0.0
                )
                wandb.log(
                    {
                        "reward/episode": episode_reward,
                        "loss/policy": update_stats.get("policy_loss", 0),
                        "loss/value": update_stats.get("value_loss", 0),
                        "curriculum/stage": curriculum.current_stage_idx
                        if curriculum
                        else 0,
                        "env/steps": total_timesteps,
                        "detection/confidence": avg_detection_confidence,
                        # GNN Communication Metrics
                        "gnn/coordination_score": avg_coordination_score,
                        "gnn/enabled": 1.0 if swarm_gnn is not None else 0.0,
                    }
                )

                # Log role distribution if available
                if gnn_role_distribution is not None:
                    role_names = ["scout", "tracker", "interceptor", "support"]
                    for i, role_name in enumerate(role_names):
                        if i < len(gnn_role_distribution):
                            wandb.log({f"gnn/role_{role_name}": float(gnn_role_distribution[i])})

            episodes += 1
            pbar.update(1)
            pbar.set_postfix({
                "reward": f"{episode_reward:.2f}",
                "stage": curriculum.stage_name if curriculum else "Default",
                "det_conf": f"{avg_detection_confidence:.2f}",
                "coord": f"{avg_coordination_score:.2f}" if swarm_gnn else "N/A",
            })

            # Curriculum Update
            if curriculum:
                # Assume a simple success metric for baseline test (e.g., reward > 50)
                success = episode_reward > 50
                res = curriculum.update(success, episode_reward)
                if res["stage_changed"]:
                    logger.info(f"📈 Promoting to Stage: {res.get('new_stage_name')}")
                    env = create_env_from_hydra(cfg, curriculum)

    except KeyboardInterrupt:
        logger.info("🛑 Training interrupted by user.")

    except Exception as e:
        logger.error(f"❌ Error occurred: {e}")
        raise e

    finally:
        pbar.close()

        # Save Final Model
        if hasattr(trainer, "save"):
            final_path = "final_model.pt"
            trainer.save(final_path)

            # Log Model Artifact to W&B
            artifact = wandb.Artifact(f"model-{cfg.experiment_name}", type="model")
            artifact.add_file(final_path)
            run.log_artifact(artifact)

        wandb.finish()
        logger.info("✅ Run Complete.")


if __name__ == "__main__":
    main()
