"""
RoboTwin Task Definitions

This module contains all manipulation task implementations.
Each task inherits from Base_Task and implements:
- setup_demo(): Initialize task-specific actors
- load_actors(): Load objects into the scene  
- play_once(): Execute the demonstration
- check_success(): Verify task completion
"""

from .._base_task import Base_Task
from ..utils import *

# Import all tasks
from .adjust_bottle import adjust_bottle
from .beat_block_hammer import beat_block_hammer
from .blocks_ranking_rgb import blocks_ranking_rgb
from .blocks_ranking_size import blocks_ranking_size
from .click_alarmclock import click_alarmclock
from .click_bell import click_bell
from .dump_bin_bigbin import dump_bin_bigbin
from .exploration import exploration
from .explore_2cabinets import explore_2cabinets
from .explore_4cabinets import explore_4cabinets
from .explore_cabinet import explore_cabinet
from .explore_cabinets import explore_cabinets
from .explore_cup import explore_cup
from .grab_roller import grab_roller
from .handover_block import handover_block
from .handover_mic import handover_mic
from .hanging_mug import hanging_mug
from .lift_pot import lift_pot
from .move_can_pot import move_can_pot
from .move_pillbottle_pad import move_pillbottle_pad
from .move_playingcard_away import move_playingcard_away
from .move_stapler_pad import move_stapler_pad
from .open_laptop import open_laptop
from .open_microwave import open_microwave
from .pick_diverse_bottles import pick_diverse_bottles
from .pick_dual_bottles import pick_dual_bottles
from .place_a2b_left import place_a2b_left
from .place_a2b_right import place_a2b_right
from .place_bread_basket import place_bread_basket
from .place_bread_skillet import place_bread_skillet
from .place_burger_fries import place_burger_fries
from .place_can_basket import place_can_basket
from .place_cans_plasticbox import place_cans_plasticbox
from .place_container_plate import place_container_plate
from .place_dual_shoes import place_dual_shoes
from .place_empty_cup import place_empty_cup
from .place_fan import place_fan
from .place_mouse_pad import place_mouse_pad
from .place_object_basket import place_object_basket
from .place_object_scale import place_object_scale
from .place_object_stand import place_object_stand
from .place_phone_stand import place_phone_stand
from .place_shoe import place_shoe
from .press_stapler import press_stapler
from .put_bottles_dustbin import put_bottles_dustbin
from .put_object_cabinet import put_object_cabinet
from .rotate_qrcode import rotate_qrcode
from .scan_object import scan_object
from .shake_bottle import shake_bottle
from .shake_bottle_horizontally import shake_bottle_horizontally
from .stack_blocks_three import stack_blocks_three
from .stack_blocks_two import stack_blocks_two
from .stack_bowls_three import stack_bowls_three
from .stack_bowls_two import stack_bowls_two
from .stamp_seal import stamp_seal
from .turn_switch import turn_switch
from .random_exploration import random_exploration

# Task registry for easy access
TASK_REGISTRY = {
    "adjust_bottle": adjust_bottle,
    "beat_block_hammer": beat_block_hammer,
    "blocks_ranking_rgb": blocks_ranking_rgb,
    "blocks_ranking_size": blocks_ranking_size,
    "click_alarmclock": click_alarmclock,
    "click_bell": click_bell,
    "dump_bin_bigbin": dump_bin_bigbin,
    "exploration": exploration,
    "explore_2cabinets": explore_2cabinets,
    "explore_4cabinets": explore_4cabinets,
    "explore_cabinet": explore_cabinet,
    "explore_cabinets": explore_cabinets,
    "explore_cup": explore_cup,
    "grab_roller": grab_roller,
    "handover_block": handover_block,
    "handover_mic": handover_mic,
    "hanging_mug": hanging_mug,
    "lift_pot": lift_pot,
    "move_can_pot": move_can_pot,
    "move_pillbottle_pad": move_pillbottle_pad,
    "move_playingcard_away": move_playingcard_away,
    "move_stapler_pad": move_stapler_pad,
    "open_laptop": open_laptop,
    "open_microwave": open_microwave,
    "pick_diverse_bottles": pick_diverse_bottles,
    "pick_dual_bottles": pick_dual_bottles,
    "place_a2b_left": place_a2b_left,
    "place_a2b_right": place_a2b_right,
    "place_bread_basket": place_bread_basket,
    "place_bread_skillet": place_bread_skillet,
    "place_burger_fries": place_burger_fries,
    "place_can_basket": place_can_basket,
    "place_cans_plasticbox": place_cans_plasticbox,
    "place_container_plate": place_container_plate,
    "place_dual_shoes": place_dual_shoes,
    "place_empty_cup": place_empty_cup,
    "place_fan": place_fan,
    "place_mouse_pad": place_mouse_pad,
    "place_object_basket": place_object_basket,
    "place_object_scale": place_object_scale,
    "place_object_stand": place_object_stand,
    "place_phone_stand": place_phone_stand,
    "place_shoe": place_shoe,
    "press_stapler": press_stapler,
    "put_bottles_dustbin": put_bottles_dustbin,
    "put_object_cabinet": put_object_cabinet,
    "random_exploration": random_exploration,
    "rotate_qrcode": rotate_qrcode,
    "scan_object": scan_object,
    "shake_bottle": shake_bottle,
    "shake_bottle_horizontally": shake_bottle_horizontally,
    "stack_blocks_three": stack_blocks_three,
    "stack_blocks_two": stack_blocks_two,
    "stack_bowls_three": stack_bowls_three,
    "stack_bowls_two": stack_bowls_two,
    "stamp_seal": stamp_seal,
    "turn_switch": turn_switch,
}


def get_task_class(task_name: str):
    """Get task class by name."""
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[task_name]


def list_tasks():
    """List all available tasks."""
    return list(TASK_REGISTRY.keys())
