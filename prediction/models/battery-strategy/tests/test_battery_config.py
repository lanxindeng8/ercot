"""
Unit tests for battery_config module
"""

import pytest
from src.battery_config import BatteryConfig, create_default_battery


class TestBatteryConfig:
    """Test BatteryConfig class"""

    def test_valid_config(self):
        """Test creating valid battery configuration"""
        config = BatteryConfig(
            E_max=10.0,
            P_ch_max=2.5,
            P_dis_max=2.5,
            eta_ch=0.95,
            eta_dis=0.95,
            SoC_min=0.1,
            SoC_max=0.9,
            SoC_0=0.5,
            SoC_T_target=0.5
        )

        assert config.E_max == 10.0
        assert config.P_ch_max == 2.5
        assert config.eta_ch == 0.95

    def test_invalid_energy_capacity(self):
        """Test that negative energy capacity raises error"""
        with pytest.raises(ValueError, match="E_max must be positive"):
            BatteryConfig(
                E_max=-10.0,
                P_ch_max=2.5,
                P_dis_max=2.5
            )

    def test_invalid_efficiency(self):
        """Test that invalid efficiency raises error"""
        with pytest.raises(ValueError, match="eta_ch must be in"):
            BatteryConfig(
                E_max=10.0,
                P_ch_max=2.5,
                P_dis_max=2.5,
                eta_ch=1.5  # Invalid: > 1
            )

    def test_invalid_soc_bounds(self):
        """Test that invalid SoC bounds raise error"""
        with pytest.raises(ValueError, match="SoC bounds"):
            BatteryConfig(
                E_max=10.0,
                P_ch_max=2.5,
                P_dis_max=2.5,
                SoC_min=0.9,
                SoC_max=0.1  # Invalid: min > max
            )

    def test_invalid_initial_soc(self):
        """Test that initial SoC outside bounds raises error"""
        with pytest.raises(ValueError, match="SoC_0 must be in"):
            BatteryConfig(
                E_max=10.0,
                P_ch_max=2.5,
                P_dis_max=2.5,
                SoC_min=0.1,
                SoC_max=0.9,
                SoC_0=1.5  # Invalid: outside bounds
            )

    def test_create_default_battery(self):
        """Test creating default battery configuration"""
        config = create_default_battery()

        assert config.E_max > 0
        assert config.P_ch_max > 0
        assert config.P_dis_max > 0
        assert 0 < config.eta_ch <= 1
        assert 0 < config.eta_dis <= 1
        assert config.SoC_min < config.SoC_max
