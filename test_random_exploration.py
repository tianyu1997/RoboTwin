#!/usr/bin/env python3
"""
Quick test script for random_exploration task.
Run with: python test_random_exploration.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_import():
    """Test that the module can be imported."""
    try:
        from envs.tasks import random_exploration as task_module
        print("✓ Module import successful")
        return True
    except ImportError as e:
        print(f"✗ Module import failed: {e}")
        return False

def test_task_registry():
    """Test that the task is registered."""
    try:
        from envs.tasks import TASK_REGISTRY
        if "random_exploration" in TASK_REGISTRY:
            print("✓ Task registered in TASK_REGISTRY")
            return True
        else:
            print("✗ Task not in registry. Available tasks:", list(TASK_REGISTRY.keys())[:5], "...")
            return False
    except ImportError as e:
        print(f"✗ Could not import TASK_REGISTRY: {e}")
        return False

def test_config_file():
    """Test that the config file exists and is valid."""
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), "task_config", "random_exploration.yml")
    
    if not os.path.exists(config_path):
        print(f"✗ Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = ["task_name", "episode_num", "max_steps"]
        missing = [k for k in required_keys if k not in config]
        
        if missing:
            print(f"✗ Missing config keys: {missing}")
            return False
        
        print(f"✓ Config file valid (task_name={config['task_name']}, episodes={config['episode_num']})")
        return True
    except Exception as e:
        print(f"✗ Config parse error: {e}")
        return False

def test_collect_script():
    """Test that the collect script is valid Python."""
    script_path = os.path.join(os.path.dirname(__file__), "script", "collect_random_exploration.py")
    
    if not os.path.exists(script_path):
        print(f"✗ Collect script not found: {script_path}")
        return False
    
    try:
        with open(script_path, 'r') as f:
            code = f.read()
        compile(code, script_path, 'exec')
        print("✓ Collect script syntax valid")
        return True
    except SyntaxError as e:
        print(f"✗ Collect script syntax error: {e}")
        return False

def test_task_class():
    """Test that the task class has required methods."""
    try:
        from envs.tasks.random_exploration import random_exploration
        
        required_methods = [
            'setup_demo',
            'load_actors', 
            'play_once',
            'check_success',
            '_generate_delta_action',
            '_record_transition',
            'get_recorded_data',
            'get_jacobian_training_data',
            '_get_current_state',
            '_state_to_vector',
        ]
        
        missing = []
        for method in required_methods:
            if not hasattr(random_exploration, method):
                missing.append(method)
        
        if missing:
            print(f"✗ Missing methods: {missing}")
            return False
        
        print(f"✓ Task class has all required methods")
        return True
    except ImportError as e:
        print(f"✗ Could not import task class: {e}")
        return False

def main():
    print("=" * 50)
    print("Testing random_exploration task")
    print("=" * 50)
    
    tests = [
        ("Config File", test_config_file),
        ("Collect Script Syntax", test_collect_script),
        ("Module Import", test_import),
        ("Task Registry", test_task_registry),
        ("Task Class Methods", test_task_class),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n[{name}]")
        try:
            results.append(test_func())
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
