"""Wrapper script for W&B sweeps that properly handles config parameters.

This script is invoked by W&B sweep agents to run training with specific
hyperparameter configurations. It extracts sweep config and launches
the training script with appropriate arguments.

Features:
- Configurable timeout for training runs
- Proper error handling and logging
- Clean W&B run finalization
"""

import logging
import os
import subprocess
import sys

import wandb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Default timeout in seconds (12 hours - reasonable for sweep runs)
# Can be overridden via SWEEP_TIMEOUT_SECONDS environment variable
DEFAULT_TIMEOUT_SECONDS = 12 * 60 * 60  # 12 hours


def get_timeout() -> int:
    """Get timeout from environment variable or use default.

    Returns:
        Timeout in seconds.
    """
    env_timeout = os.environ.get("SWEEP_TIMEOUT_SECONDS")
    if env_timeout is not None:
        try:
            return int(env_timeout)
        except ValueError:
            logger.warning(
                f"Invalid SWEEP_TIMEOUT_SECONDS value: {env_timeout}, "
                f"using default: {DEFAULT_TIMEOUT_SECONDS}"
            )
    return DEFAULT_TIMEOUT_SECONDS


def main() -> int:
    """Run a single sweep training iteration.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    # Initialize wandb - this gets config from sweep
    wandb.init()

    # Get sweep config
    config = wandb.config
    timeout = get_timeout()

    # Build command with actual values from sweep config
    cmd = [
        sys.executable,
        "run_muzero_training.py",
        "--quick",
        "--small",
        "--wandb",
        f"--lr={config.lr}",
        f"--batch-size={config.batch_size}",
        f"--simulations={config.simulations}",
        f"--res-blocks={config.res_blocks}",
        f"--filters={config.filters}",
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    logger.info(f"Timeout: {timeout} seconds ({timeout / 3600:.1f} hours)")

    return_code = 1  # Default to failure
    try:
        # Run the actual training with timeout
        result = subprocess.run(
            cmd,
            capture_output=False,
            timeout=timeout,
        )
        return_code = result.returncode

        if return_code == 0:
            logger.info("Training completed successfully")
        else:
            logger.warning(f"Training exited with code: {return_code}")

    except subprocess.TimeoutExpired:
        logger.error(
            f"Training exceeded timeout of {timeout} seconds. "
            f"Consider increasing SWEEP_TIMEOUT_SECONDS environment variable."
        )
        return_code = 124  # Standard timeout exit code

    except Exception as e:
        logger.error(f"Error running training: {e}")
        return_code = 1

    finally:
        # Always clean up wandb run
        try:
            wandb.finish()
        except Exception as e:
            logger.warning(f"Error finishing W&B run: {e}")

    return return_code


if __name__ == "__main__":
    sys.exit(main())
