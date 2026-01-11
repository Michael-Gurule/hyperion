import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import os
import torch
import numpy as np
from collections import defaultdict
import logging

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

    # D. Agent / Policy (The "Brain")
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
        while (
            episodes < cfg.evaluation.episodes
        ):  # Using evaluation count for short test, usually total_timesteps
            observations, _ = env.reset()
            if intrinsic_calculator:
                intrinsic_calculator.reset()

            episode_reward = 0
            steps = 0

            # --- Rollout ---
            for step in range(cfg.hardware.rollout_fragment_length):
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

                # Intrinsic Reward Calculation
                combined_rewards = rewards.copy()
                if intrinsic_calculator and env.target_state:
                    # (Simplified for baseline test- assumes logic from enhanced script)
                    # Inject the intrinsic calculation loop here
                    pass

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
                wandb.log(
                    {
                        "reward/episode": episode_reward,
                        "loss/policy": update_stats.get("policy_loss", 0),
                        "loss/value": update_stats.get("value_loss", 0),
                        "curriculum/stage": curriculum.current_stage_idx
                        if curriculum
                        else 0,
                        "env/steps": total_timesteps,
                    }
                )

            episodes += 1
            if episodes % 10 == 0:
                logger.info(
                    f"Episode {episodes} | Reward: {episode_reward:.2f} | Stage: {curriculum.stage_name if curriculum else 'Default'}"
                )

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
