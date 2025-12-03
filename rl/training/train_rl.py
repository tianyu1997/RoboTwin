#!/usr/bin/env python3
"""
Unified RL Training Entry Point for F1-VLA

This script provides a unified entry point for all three phases of RL training:
- Phase 1 (Teacher): Train world model with random exploration
- Phase 2 (Student): Train policy using PPO with world model rewards
- Phase 3 (Adversarial): Alternating training of WM and Explorer

Usage:
    # Run Phase 1 (Teacher)
    python train_rl.py --phase teacher --output_dir outputs/teacher
    
    # Run Phase 2 (Student) with teacher checkpoint
    python train_rl.py --phase student --teacher_checkpoint outputs/teacher/checkpoint-10000
    
    # Run Phase 3 (Adversarial) with both checkpoints
    python train_rl.py --phase adversarial \
        --teacher_checkpoint outputs/teacher/checkpoint-10000 \
        --student_checkpoint outputs/student/checkpoint-10000

All training parameters can be configured via:
1. rl_config.yaml (default: RoboTwin/rl/rl_config.yaml)
2. Command-line overrides
"""

import os
import sys
import argparse
import subprocess

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified RL Training for F1-VLA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required: training phase
    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=["teacher", "student", "adversarial", "1", "2", "3"],
        help="Training phase: teacher (1), student (2), or adversarial (3)"
    )
    
    # Config
    parser.add_argument(
        "--rl_config",
        type=str,
        default=None,
        help="Path to RL config YAML file"
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default=None,
        help="Override model config file path"
    )
    
    # Checkpoints
    parser.add_argument(
        "--teacher_checkpoint",
        type=str,
        default=None,
        help="Path to teacher checkpoint (required for student/adversarial)"
    )
    parser.add_argument(
        "--student_checkpoint",
        type=str,
        default=None,
        help="Path to student checkpoint (optional for adversarial)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from checkpoint"
    )
    
    # Override parameters
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--total_iterations", type=int, default=None)
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    
    # Memory/Sequential settings
    parser.add_argument(
        "--sequential_training",
        action="store_true",
        default=None,
        help="Enable sequential training mode for memory state propagation"
    )
    parser.add_argument(
        "--no_sequential_training",
        action="store_false",
        dest="sequential_training",
        help="Disable sequential training mode"
    )
    
    # Device & debug
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=None)
    
    return parser.parse_args()


def build_command(args) -> list:
    """Build the command to run the appropriate training script."""
    # Normalize phase name
    phase_map = {
        "teacher": "teacher",
        "student": "student",
        "adversarial": "adversarial",
        "1": "teacher",
        "2": "student",
        "3": "adversarial",
    }
    phase = phase_map[args.phase]
    
    # Select script
    script_map = {
        "teacher": "train_teacher_rl.py",
        "student": "train_student_rl.py",
        "adversarial": "train_adversarial_rl.py",
    }
    script = os.path.join(SCRIPT_DIR, script_map[phase])
    
    # Build command
    cmd = [sys.executable, script]
    
    # Add arguments
    if args.rl_config:
        cmd.extend(["--rl_config", args.rl_config])
    if args.model_config:
        cmd.extend(["--model_config", args.model_config])
    
    # Checkpoints
    if args.teacher_checkpoint:
        cmd.extend(["--teacher_checkpoint", args.teacher_checkpoint])
    if args.student_checkpoint:
        cmd.extend(["--student_checkpoint", args.student_checkpoint])
    if args.resume:
        cmd.extend(["--resume", args.resume])
    
    # Override parameters
    if args.output_dir:
        cmd.extend(["--output_dir", args.output_dir])
    if args.total_iterations is not None:
        cmd.extend(["--total_iterations", str(args.total_iterations)])
    if args.num_episodes is not None:
        cmd.extend(["--num_episodes", str(args.num_episodes)])
    if args.batch_size is not None:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if args.learning_rate is not None:
        cmd.extend(["--learning_rate", str(args.learning_rate)])
    
    # Memory settings
    if args.sequential_training is not None:
        if args.sequential_training:
            cmd.append("--sequential_training")
        else:
            cmd.append("--no_sequential_training")
    
    # Device & debug
    if args.device:
        cmd.extend(["--device", args.device])
    if args.debug:
        cmd.append("--debug")
    
    return cmd


def validate_args(args):
    """Validate arguments based on phase."""
    phase_map = {
        "teacher": "teacher",
        "student": "student",
        "adversarial": "adversarial",
        "1": "teacher",
        "2": "student",
        "3": "adversarial",
    }
    phase = phase_map[args.phase]
    
    # Student and adversarial phases require teacher checkpoint
    if phase in ["student", "adversarial"] and not args.teacher_checkpoint:
        print(f"Error: Phase '{phase}' requires --teacher_checkpoint")
        print(f"Example: python train_rl.py --phase {phase} "
              f"--teacher_checkpoint outputs/teacher/checkpoint-10000")
        return False
    
    return True


def main():
    args = parse_args()
    
    # Validate arguments
    if not validate_args(args):
        sys.exit(1)
    
    # Build command
    cmd = build_command(args)
    
    # Print command
    print("=" * 70)
    print("F1-VLA RL Training")
    print("=" * 70)
    print(f"Phase: {args.phase}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 70)
    
    # Run command
    try:
        result = subprocess.run(cmd, check=True)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
