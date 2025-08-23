"""
Test imports to ensure the reorganized project structure works correctly.
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_config_imports():
    """Test that configuration can be imported."""
    try:
        # Add project root to path for config import
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import config.settings
        assert hasattr(config.settings, 'DB_PATH')
        assert hasattr(config.settings, 'STOCK_TICKERS')
        print("✅ Config imports successful")
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        raise

def test_utils_imports():
    """Test that utility modules can be imported."""
    try:
        from market_dashboard.utils import data_processing, formatting, calculations
        assert hasattr(data_processing, 'resample_data')
        assert hasattr(formatting, 'pct_change_str')
        assert hasattr(calculations, 'simple_moving_average')
        print("✅ Utils imports successful")
    except ImportError as e:
        print(f"❌ Utils import failed: {e}")
        raise

def test_data_imports():
    """Test that data modules can be imported."""
    try:
        from market_dashboard.data import DatabaseManager, DataFetcher, DataLoader
        print("✅ Data imports successful")
    except ImportError as e:
        print(f"❌ Data import failed: {e}")
        raise

def test_ai_imports():
    """Test that AI modules can be imported."""
    try:
        from market_dashboard.ai import InsightsGenerator, OllamaClient, OllamaError
        print("✅ AI imports successful")
    except ImportError as e:
        print(f"❌ AI import failed: {e}")
        raise

def test_components_imports():
    """Test that component modules can be imported."""
    try:
        from market_dashboard.components import layouts, charts, callbacks
        assert hasattr(layouts, 'create_landing_layout')
        assert hasattr(charts, 'ChartGenerator')
        assert hasattr(callbacks, 'register_callbacks')
        print("✅ Components imports successful")
    except ImportError as e:
        print(f"❌ Components import failed: {e}")
        raise

def test_app_import():
    """Test that the main app can be imported."""
    try:
        from market_dashboard.app import create_app
        print("✅ App import successful")
    except ImportError as e:
        print(f"❌ App import failed: {e}")
        raise

def run_all_tests():
    """Run all import tests."""
    print("Running import tests for reorganized project...")
    
    tests = [
        test_config_imports,
        test_utils_imports,
        test_data_imports,
        test_ai_imports,
        test_components_imports,
        test_app_import
    ]
    
    failed_tests = []
    
    for test in tests:
        try:
            test()
        except Exception as e:
            failed_tests.append((test.__name__, str(e)))
    
    if failed_tests:
        print(f"\n❌ {len(failed_tests)} tests failed:")
        for test_name, error in failed_tests:
            print(f"  - {test_name}: {error}")
        return False
    else:
        print(f"\n✅ All {len(tests)} import tests passed!")
        return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
