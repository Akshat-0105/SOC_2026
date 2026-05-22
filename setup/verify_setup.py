"""
Setup verification script for beginners.
Run this after installing requirements to check everything works.

Usage:
    python verify_setup.py
"""

import sys

def check_python_version():
    """Check Python version is compatible."""
    print("Checking Python version...")
    major, minor = sys.version_info[:2]

    if major < 3 or (major == 3 and minor < 9):
        print(f"❌ Python {major}.{minor} detected. Need Python 3.9+")
        return False

    print(f"✅ Python {major}.{minor} is compatible")
    return True


def check_pytorch():
    """Check PyTorch installation."""
    print("\nChecking PyTorch...")
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} installed")

        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"✅ CUDA available! Device: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
        else:
            print("⚠️  CUDA not available (CPU only)")
            print("   Training will be slower but will work fine")

        # Simple tensor test
        x = torch.randn(3, 3)
        y = x + 2
        assert y.shape == (3, 3)
        print("✅ PyTorch operations working")

        return True
    except ImportError:
        print("❌ PyTorch not installed")
        print("   Install with: pip install torch torchvision")
        return False
    except Exception as e:
        print(f"❌ PyTorch error: {e}")
        return False


def check_gymnasium():
    """Check Gymnasium (OpenAI Gym successor) installation."""
    print("\nChecking Gymnasium...")
    try:
        import gymnasium as gym
        print(f"✅ Gymnasium installed")

        # Test creating an environment
        env = gym.make("CartPole-v1")
        obs, _ = env.reset()
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.close()

        print("✅ CartPole environment working")
        return True
    except ImportError:
        print("❌ Gymnasium not installed")
        print("   Install with: pip install gymnasium")
        return False
    except Exception as e:
        print(f"❌ Gymnasium error: {e}")
        return False


def check_numpy():
    """Check NumPy installation."""
    print("\nChecking NumPy...")
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__} installed")

        # Simple test
        arr = np.random.randn(10, 10)
        assert arr.shape == (10, 10)
        print("✅ NumPy operations working")

        return True
    except ImportError:
        print("❌ NumPy not installed")
        print("   Install with: pip install numpy")
        return False
    except Exception as e:
        print(f"❌ NumPy error: {e}")
        return False


def check_pandas():
    """Check Pandas installation."""
    print("\nChecking Pandas...")
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__} installed")

        # Simple test
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        assert len(df) == 3
        print("✅ Pandas operations working")

        return True
    except ImportError:
        print("❌ Pandas not installed")
        print("   Install with: pip install pandas")
        return False
    except Exception as e:
        print(f"❌ Pandas error: {e}")
        return False


def check_matplotlib():
    """Check Matplotlib installation."""
    print("\nChecking Matplotlib...")
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        print(f"✅ Matplotlib {matplotlib.__version__} installed")

        # Simple plot test (don't show)
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        plt.close(fig)
        print("✅ Matplotlib plotting working")

        return True
    except ImportError:
        print("❌ Matplotlib not installed")
        print("   Install with: pip install matplotlib")
        return False
    except Exception as e:
        print(f"❌ Matplotlib error: {e}")
        return False


def check_optional_libraries():
    """Check optional but recommended libraries."""
    print("\n=== Optional Libraries ===")

    optional = {
        'wandb': 'Weights & Biases (experiment tracking)',
        'tqdm': 'Progress bars',
        'seaborn': 'Advanced plotting',
        'pettingzoo': 'Multi-agent environments',
        'hmmlearn': 'Hidden Markov Models',
        'ruptures': 'Change-point detection'
    }

    for lib, description in optional.items():
        try:
            __import__(lib)
            print(f"✅ {lib} - {description}")
        except ImportError:
            print(f"⚠️  {lib} not installed - {description}")


def run_simple_training_test():
    """Run a very simple training test to verify everything works together."""
    print("\n=== Running Simple Training Test ===")
    print("Training a dummy agent for 100 steps...")

    try:
        import torch
        import torch.nn as nn
        import gymnasium as gym
        import numpy as np

        # Simple policy network
        class SimplePolicy(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 2)

            def forward(self, x):
                return self.fc(x)

        # Create environment
        env = gym.make("CartPole-v1")
        policy = SimplePolicy()
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        # Mini training loop
        obs, _ = env.reset()
        total_reward = 0

        for step in range(100):
            # Forward pass
            obs_tensor = torch.FloatTensor(obs)
            logits = policy(obs_tensor)
            action = torch.argmax(logits).item()

            # Environment step
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward

            if done or truncated:
                obs, _ = env.reset()
                total_reward = 0

            # Dummy backward pass (just to test)
            if step % 10 == 0:
                loss = torch.tensor(1.0, requires_grad=True)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        env.close()
        print("✅ Simple training loop works!")
        print(f"   Agent collected rewards: {total_reward:.1f}")
        return True

    except Exception as e:
        print(f"❌ Training test failed: {e}")
        return False


def print_summary(results):
    """Print summary of checks."""
    print("\n" + "="*50)
    print("SETUP VERIFICATION SUMMARY")
    print("="*50)

    core_libs = ['python', 'pytorch', 'gymnasium', 'numpy', 'pandas', 'matplotlib']
    core_results = [results.get(lib, False) for lib in core_libs]

    if all(core_results):
        print("✅ All core libraries installed successfully!")
        print("✅ You're ready to start Week 1!")
        print("\nNext steps:")
        print("  1. Read GETTING_STARTED.md")
        print("  2. Go to week1/README.md")
        print("  3. Start with Assignment 1")
    else:
        print("❌ Some core libraries are missing or have issues")
        print("\nPlease fix the issues above before proceeding.")
        print("Check setup/setup_guide.md for detailed instructions.")

    print("\nOptional libraries:")
    print("  These can be installed later as needed")
    print("="*50)


def main():
    """Run all checks."""
    print("="*50)
    print("SETUP VERIFICATION SCRIPT")
    print("="*50)
    print("This will check if your environment is set up correctly.\n")

    results = {}

    # Core checks
    results['python'] = check_python_version()
    results['pytorch'] = check_pytorch()
    results['gymnasium'] = check_gymnasium()
    results['numpy'] = check_numpy()
    results['pandas'] = check_pandas()
    results['matplotlib'] = check_matplotlib()

    # Optional libraries
    check_optional_libraries()

    # Integration test
    results['training_test'] = run_simple_training_test()

    # Summary
    print_summary(results)

    # Return exit code
    core_passed = all([
        results['python'],
        results['pytorch'],
        results['gymnasium'],
        results['numpy'],
        results['pandas'],
        results['matplotlib']
    ])

    sys.exit(0 if core_passed else 1)


if __name__ == '__main__':
    main()
