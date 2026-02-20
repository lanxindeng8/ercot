"""
Battery Energy Storage System (BESS) Configuration

This module defines the battery parameters and configuration for optimization.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BatteryConfig:
    """
    Battery Energy Storage System parameters

    Attributes:
        E_max: Battery energy capacity (MWh)
        P_ch_max: Maximum charging power (MW)
        P_dis_max: Maximum discharging power (MW)
        eta_ch: Charging efficiency (0-1)
        eta_dis: Discharging efficiency (0-1)
        SoC_min: Minimum state of charge (0-1)
        SoC_max: Maximum state of charge (0-1)
        SoC_0: Initial state of charge (0-1)
        SoC_T_target: Target end state of charge (0-1), optional
        delta_t: Time resolution in hours (5 min = 5/60 hours)
        P_export_max: Max grid export power (MW)
        P_import_max: Max grid import power (MW)
        c_deg: Linear degradation cost coefficient ($/MWh_dis)
        lambda_delta_p: Ramp penalty coefficient ($/MW)
    """
    E_max: float  # MWh
    P_ch_max: float  # MW
    P_dis_max: float  # MW
    eta_ch: float = 0.95  # Default 95% efficiency
    eta_dis: float = 0.95  # Default 95% efficiency
    SoC_min: float = 0.1  # Default 10% minimum
    SoC_max: float = 0.9  # Default 90% maximum
    SoC_0: float = 0.5  # Default 50% initial
    SoC_T_target: Optional[float] = None  # None means no end constraint
    delta_t: float = 5.0 / 60.0  # 5 minutes in hours
    P_export_max: Optional[float] = None  # MW, grid export limit
    P_import_max: Optional[float] = None  # MW, grid import limit
    c_deg: float = 10.0  # $/MWh_dis
    lambda_delta_p: float = 1.0  # $/MW

    def __post_init__(self):
        """Validate battery parameters"""
        self._validate()

    def _validate(self):
        """Validate all parameters are within acceptable ranges"""
        if self.E_max <= 0:
            raise ValueError(f"E_max must be positive, got {self.E_max}")
        if self.P_ch_max <= 0:
            raise ValueError(f"P_ch_max must be positive, got {self.P_ch_max}")
        if self.P_dis_max <= 0:
            raise ValueError(f"P_dis_max must be positive, got {self.P_dis_max}")

        if not 0 < self.eta_ch <= 1:
            raise ValueError(f"eta_ch must be in (0, 1], got {self.eta_ch}")
        if not 0 < self.eta_dis <= 1:
            raise ValueError(f"eta_dis must be in (0, 1], got {self.eta_dis}")

        if not 0 <= self.SoC_min < self.SoC_max <= 1:
            raise ValueError(
                f"SoC bounds must satisfy 0 <= SoC_min < SoC_max <= 1, "
                f"got SoC_min={self.SoC_min}, SoC_max={self.SoC_max}"
            )

        if not self.SoC_min <= self.SoC_0 <= self.SoC_max:
            raise ValueError(
                f"SoC_0 must be in [SoC_min, SoC_max], "
                f"got SoC_0={self.SoC_0}, SoC_min={self.SoC_min}, SoC_max={self.SoC_max}"
            )

        if self.SoC_T_target is not None:
            if not self.SoC_min <= self.SoC_T_target <= self.SoC_max:
                raise ValueError(
                    f"SoC_T_target must be in [SoC_min, SoC_max], "
                    f"got SoC_T_target={self.SoC_T_target}"
                )

        if self.delta_t <= 0:
            raise ValueError(f"delta_t must be positive, got {self.delta_t}")

        if self.P_export_max is not None and self.P_export_max <= 0:
            raise ValueError(f"P_export_max must be positive when set, got {self.P_export_max}")

        if self.P_import_max is not None and self.P_import_max <= 0:
            raise ValueError(f"P_import_max must be positive when set, got {self.P_import_max}")

        if self.c_deg < 0:
            raise ValueError(f"c_deg must be non-negative, got {self.c_deg}")

        if self.lambda_delta_p < 0:
            raise ValueError(f"lambda_delta_p must be non-negative, got {self.lambda_delta_p}")


def create_default_battery() -> BatteryConfig:
    """
    Create a default battery configuration for testing

    Returns:
        BatteryConfig with typical commercial battery parameters
    """
    return BatteryConfig(
        E_max=10.0,  # 10 MWh capacity
        P_ch_max=2.5,  # 2.5 MW charging power
        P_dis_max=2.5,  # 2.5 MW discharging power
        eta_ch=0.95,
        eta_dis=0.95,
        SoC_min=0.1,
        SoC_max=0.9,
        SoC_0=0.5,
        SoC_T_target=None,  # Return to initial SoC
        delta_t=5.0 / 60.0,  # 5 minutes
        P_export_max=2.5,
        P_import_max=2.5,
        c_deg=10.0,
        lambda_delta_p=1.0
    )
