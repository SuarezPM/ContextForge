import pytest
import threading
import time
from unittest.mock import patch
import apohara_context_forge.metrics.vram_monitor as vram_module

def test_get_monitor_singleton():
    """Test that get_monitor() returns the same VRAMMonitor instance."""
    # Reset singleton state
    vram_module._monitor = None

    m1 = vram_module.get_monitor()
    m2 = vram_module.get_monitor()

    assert isinstance(m1, vram_module.VRAMMonitor)
    assert m1 is m2

def test_get_monitor_concurrent_access():
    """Test that concurrent access to get_monitor() initializes the singleton safely."""
    # Reset singleton state
    vram_module._monitor = None

    # We want to force a race condition window by pausing during initialization
    original_init = vram_module.VRAMMonitor.__init__

    init_calls = 0

    def slow_init(self, *args, **kwargs):
        nonlocal init_calls
        init_calls += 1
        time.sleep(0.1) # Simulate slow initialization
        original_init(self, *args, **kwargs)

    with patch.object(vram_module.VRAMMonitor, '__init__', side_effect=slow_init, autospec=True):
        monitors = []
        def fetch_monitor():
            monitors.append(vram_module.get_monitor())

        threads = [threading.Thread(target=fetch_monitor) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should have received the exact same instance
        first_monitor = monitors[0]
        for m in monitors[1:]:
            assert m is first_monitor

        # Initialization should only happen exactly once
        assert init_calls == 1
