import pytest
from unittest.mock import patch

import apohara_context_forge.metrics.prometheus_metrics as prom_mod

def test_record_vram_metrics_known_mode():
    with patch.object(prom_mod, "vram_pressure_ratio") as mock_pressure, \
         patch.object(prom_mod, "vram_used_gb") as mock_used, \
         patch.object(prom_mod, "vram_available_gb") as mock_avail, \
         patch.object(prom_mod, "eviction_mode") as mock_mode, \
         patch("apohara_context_forge.metrics.prometheus_metrics.PROMETHEUS_AVAILABLE", True, create=True), \
         patch("apohara_context_forge.metrics.prometheus_metrics._ENABLED", True, create=True):

        prom_mod.record_vram_metrics(pressure=0.85, used_gb=20.0, available_gb=4.0, mode="critical")

        mock_pressure.set.assert_called_once_with(0.85)
        mock_used.set.assert_called_once_with(20.0)
        mock_avail.set.assert_called_once_with(4.0)
        mock_mode.set.assert_called_once_with(3)  # critical = 3

def test_record_vram_metrics_unknown_mode():
    with patch.object(prom_mod, "vram_pressure_ratio") as mock_pressure, \
         patch.object(prom_mod, "vram_used_gb") as mock_used, \
         patch.object(prom_mod, "vram_available_gb") as mock_avail, \
         patch.object(prom_mod, "eviction_mode") as mock_mode, \
         patch("apohara_context_forge.metrics.prometheus_metrics.PROMETHEUS_AVAILABLE", True, create=True), \
         patch("apohara_context_forge.metrics.prometheus_metrics._ENABLED", True, create=True):

        prom_mod.record_vram_metrics(pressure=0.5, used_gb=10.0, available_gb=14.0, mode="unknown_mode")

        mock_pressure.set.assert_called_once_with(0.5)
        mock_used.set.assert_called_once_with(10.0)
        mock_avail.set.assert_called_once_with(14.0)
        mock_mode.set.assert_called_once_with(0)  # default = 0
