# test_phase0.py
print("=== Testing Phase 0 Communication ===\n")

# Test 1: Import all core modules
print("1. Testing imports...")
try:
    from wednesday.main import Wednesday
    from wednesday.config import Config
    from wednesday.constants import DEFAULT_PERSONALITY
    from wednesday.exceptions import WednesdayError
    print("   ✓ All modules imported successfully")
except Exception as e:
    print(f"   ✗ Import failed: {e}")

# Test 2: Test Config -> Wednesday communication
print("\n2. Testing Config -> Wednesday...")
try:
    config = Config()
    wed = Wednesday(config)
    print("   ✓ Config passed to Wednesday successfully")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 3: Test constants being used
print("\n3. Testing Constants access...")
try:
    print(f"   ✓ Personality traits: {list(DEFAULT_PERSONALITY.keys())}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 4: Test error handling
print("\n4. Testing Exceptions...")
try:
    raise WednesdayError("Test error")
except WednesdayError as e:
    print(f"   ✓ Error raised and caught: {e}")

# Test 5: Test full pipeline
print("\n5. Testing full Wednesday pipeline...")
try:
    wed = Wednesday()
    wed.initialize()
    response = wed.process_input("Hello")
    print(f"   ✓ Wednesday responded: '{response}'")
    status = wed.get_status()
    print(f"   ✓ Status retrieved: {status['name']} v{status['version']}")
    wed.shutdown()
    print("   ✓ Shutdown successful")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("\n=== Test Complete ===")