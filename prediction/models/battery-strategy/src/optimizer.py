"""
Battery Energy Storage System Optimization using Linear Programming

This module implements the LP-based optimization strategy for battery
charging/discharging based on electricity price arbitrage.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pulp

from .battery_config import BatteryConfig


@dataclass
class OptimizationResult:
    """
    Results from battery optimization

    Attributes:
        P_ch: Charging power schedule (MW) for each time step
        P_dis: Discharging power schedule (MW) for each time step
        SoC: State of charge schedule (0-1) for each time step
        ramp_up: Positive part of net power change (MW) per step
        ramp_down: Negative part of net power change (MW) per step
        total_revenue: Total revenue from arbitrage ($)
        status: Optimization status (Optimal, Infeasible, etc.)
        solve_time: Time taken to solve the optimization (seconds)
    """
    P_ch: np.ndarray
    P_dis: np.ndarray
    SoC: np.ndarray
    ramp_up: np.ndarray
    ramp_down: np.ndarray
    total_revenue: float
    status: str
    solve_time: float

    def get_net_power(self) -> np.ndarray:
        """
        Get net power (positive = discharging, negative = charging)

        Returns:
            Net power schedule (MW)
        """
        return self.P_dis - self.P_ch

    def get_energy_throughput(self, delta_t: float) -> float:
        """
        Calculate total energy throughput

        Args:
            delta_t: Time step duration in hours

        Returns:
            Total energy throughput (MWh)
        """
        return np.sum(self.P_ch + self.P_dis) * delta_t


@dataclass
class LPInput:
    """
    Explicit LP inputs for Step1 (5-minute granularity, pure LP).

    Attributes mirror BatteryConfig plus price series to keep the solver
    interface explicit and testable.
    """

    prices: np.ndarray  # $/MWh, length T
    E_max: float  # MWh
    P_ch_max: float  # MW
    P_dis_max: float  # MW
    eta_ch: float
    eta_dis: float
    SoC_min: float
    SoC_max: float
    SoC_0: float
    SoC_T_target: Optional[float]
    delta_t: float  # hours (5min = 1/12)
    P_export_max: Optional[float] = None  # MW
    P_import_max: Optional[float] = None  # MW
    c_deg: float = 0.0  # $/MWh_dis
    lambda_delta_p: float = 0.0  # $/MW

    def __post_init__(self):
        self.prices = np.asarray(self.prices, dtype=float)
        if len(self.prices) <= 0:
            raise ValueError(f"Price array must have positive length, got {len(self.prices)}")
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
        if self.c_deg < 0:
            raise ValueError(f"c_deg must be non-negative, got {self.c_deg}")
        if self.lambda_delta_p < 0:
            raise ValueError(f"lambda_delta_p must be non-negative, got {self.lambda_delta_p}")
        if self.P_export_max is not None and self.P_export_max <= 0:
            raise ValueError(f"P_export_max must be positive when set, got {self.P_export_max}")
        if self.P_import_max is not None and self.P_import_max <= 0:
            raise ValueError(f"P_import_max must be positive when set, got {self.P_import_max}")


def create_lp_input(prices: np.ndarray, config: BatteryConfig) -> LPInput:
    """
    Construct LPInput from price series and battery configuration.
    """
    return LPInput(
        prices=prices,
        E_max=config.E_max,
        P_ch_max=config.P_ch_max,
        P_dis_max=config.P_dis_max,
        eta_ch=config.eta_ch,
        eta_dis=config.eta_dis,
        SoC_min=config.SoC_min,
        SoC_max=config.SoC_max,
        SoC_0=config.SoC_0,
        SoC_T_target=config.SoC_T_target,
        delta_t=config.delta_t,
        P_export_max=config.P_export_max,
        P_import_max=config.P_import_max,
        c_deg=config.c_deg,
        lambda_delta_p=config.lambda_delta_p,
    )


class BatteryOptimizer:
    """
    Linear Programming optimizer for battery energy storage system

    Uses PuLP library to formulate and solve the LP problem for maximizing
    revenue from price arbitrage.
    """

    def __init__(self, config: BatteryConfig):
        """
        Initialize optimizer with battery configuration

        Args:
            config: Battery configuration parameters
        """
        self.config = config

    def optimize(
        self,
        prices: np.ndarray,
        solver_name: str = "PULP_CBC_CMD",
        verbose: bool = False
    ) -> OptimizationResult:
        """
        Optimize battery charging/discharging schedule

        Args:
            prices: Electricity prices ($/MWh) for each 5-minute interval
            solver_name: PuLP solver to use (default: CBC)
            verbose: Whether to print solver output

        Returns:
            OptimizationResult with optimal schedule and revenue

        Raises:
            ValueError: If price array length is invalid
        """
        lp_input = create_lp_input(prices, self.config)
        return self.build_LP(lp_input, solver_name=solver_name, verbose=verbose)

    def build_LP(
        self,
        lp_input: LPInput,
        solver_name: str = "PULP_CBC_CMD",
        verbose: bool = False
    ) -> OptimizationResult:
        """
        Build and solve the LP defined in BESS_v2.3 (115-117),
        including degradation and ramp penalties.
        """
        T = len(lp_input.prices)

        # Create LP problem
        prob = pulp.LpProblem("Battery_Arbitrage", pulp.LpMaximize)

        # Decision variables
        P_ch = [
            pulp.LpVariable(f"P_ch_{t}", lowBound=0, upBound=lp_input.P_ch_max)
            for t in range(T)
        ]
        P_dis = [
            pulp.LpVariable(f"P_dis_{t}", lowBound=0, upBound=lp_input.P_dis_max)
            for t in range(T)
        ]
        SoC = [
            pulp.LpVariable(
                f"SoC_{t}",
                lowBound=lp_input.SoC_min,
                upBound=lp_input.SoC_max
            )
            for t in range(T)
        ]

        # Ramp linearization vars (u: positive ΔP, v: negative ΔP)
        ramp_up = [pulp.LpVariable(f"u_{t}", lowBound=0) for t in range(T)]
        ramp_down = [pulp.LpVariable(f"v_{t}", lowBound=0) for t in range(T)]

        # Objective function: maximize revenue
        # Revenue = price spread - degradation - ramp penalty
        revenue = pulp.lpSum(
            [
                lp_input.prices[t] * (P_dis[t] - P_ch[t]) * lp_input.delta_t
                - lp_input.c_deg * P_dis[t] * lp_input.delta_t
                - lp_input.lambda_delta_p * (ramp_up[t] + ramp_down[t])
                for t in range(T)
            ]
        )
        prob += revenue

        # Constraints

        # (1) SoC dynamics - first time step
        prob += (
            SoC[0] == lp_input.SoC_0 +
            (lp_input.eta_ch * P_ch[0] * lp_input.delta_t) / lp_input.E_max -
            (P_dis[0] * lp_input.delta_t) / (lp_input.eta_dis * lp_input.E_max)
        ), "SoC_dynamics_0"

        # (1) SoC dynamics - remaining time steps
        for t in range(1, T):
            prob += (
                SoC[t] == SoC[t-1] +
                (lp_input.eta_ch * P_ch[t] * lp_input.delta_t) / lp_input.E_max -
                (P_dis[t] * lp_input.delta_t) / (lp_input.eta_dis * lp_input.E_max)
            ), f"SoC_dynamics_{t}"

        # (4) Terminal SoC constraint (if specified)
        if lp_input.SoC_T_target is not None:
            prob += SoC[T-1] == lp_input.SoC_T_target, "Terminal_SoC"

        # (5) Ramp / ΔP linearization
        # Net power at step t: P_dis - P_ch
        prob += ramp_up[0] == 0, "Ramp_up_init"
        prob += ramp_down[0] == 0, "Ramp_down_init"
        for t in range(1, T):
            net_diff = (P_dis[t] - P_ch[t]) - (P_dis[t-1] - P_ch[t-1])
            prob += ramp_up[t] >= net_diff, f"Ramp_up_pos_{t}"
            prob += ramp_down[t] >= -net_diff, f"Ramp_down_neg_{t}"

        # (6) Grid (PCC) export/import limits if provided
        for t in range(T):
            if lp_input.P_export_max is not None:
                prob += (P_dis[t] - P_ch[t]) <= lp_input.P_export_max, f"P_export_max_{t}"
            if lp_input.P_import_max is not None:
                prob += (P_ch[t] - P_dis[t]) <= lp_input.P_import_max, f"P_import_max_{t}"

        # Solve the problem
        solver = self._get_solver(solver_name, verbose)
        solve_start = pulp.clock()
        prob.solve(solver)
        solve_time = pulp.clock() - solve_start

        # Extract results
        P_ch_result = np.array([pulp.value(P_ch[t]) for t in range(T)])
        P_dis_result = np.array([pulp.value(P_dis[t]) for t in range(T)])
        SoC_result = np.array([pulp.value(SoC[t]) for t in range(T)])
        ramp_up_result = np.array([pulp.value(ramp_up[t]) for t in range(T)])
        ramp_down_result = np.array([pulp.value(ramp_down[t]) for t in range(T)])
        total_revenue = pulp.value(prob.objective)
        status_str = pulp.LpStatus[prob.status]

        # If not optimal, return NaNs to signal caller to inspect status
        if status_str != "Optimal" or total_revenue is None:
            P_ch_result = np.full(T, np.nan)
            P_dis_result = np.full(T, np.nan)
            SoC_result = np.full(T, np.nan)
            ramp_up_result = np.full(T, np.nan)
            ramp_down_result = np.full(T, np.nan)
            total_revenue = np.nan

        return OptimizationResult(
            P_ch=P_ch_result,
            P_dis=P_dis_result,
            SoC=SoC_result,
            ramp_up=ramp_up_result,
            ramp_down=ramp_down_result,
            total_revenue=total_revenue,
            status=status_str,
            solve_time=solve_time
        )

    def _get_solver(self, solver_name: str, verbose: bool):
        """
        Get PuLP solver instance

        Args:
            solver_name: Name of solver to use
            verbose: Whether to show solver output

        Returns:
            PuLP solver instance
        """
        msg = 1 if verbose else 0

        if solver_name == "PULP_CBC_CMD":
            return pulp.PULP_CBC_CMD(msg=msg)
        elif solver_name == "GUROBI":
            return pulp.GUROBI(msg=msg)
        elif solver_name == "CPLEX":
            return pulp.CPLEX_CMD(msg=msg)
        elif solver_name == "GLPK":
            return pulp.GLPK_CMD(msg=msg)
        else:
            # Default solver
            return pulp.PULP_CBC_CMD(msg=msg)


def interpolate_hourly_to_5min(hourly_prices: np.ndarray) -> np.ndarray:
    """
    Interpolate hourly DAM prices to 5-minute RTM prices using linear interpolation

    Args:
        hourly_prices: Array of 24 hourly prices ($/MWh)

    Returns:
        Array of 288 prices at 5-minute resolution ($/MWh)

    Raises:
        ValueError: If hourly_prices length is not 24
    """
    if len(hourly_prices) != 24:
        raise ValueError(f"Expected 24 hourly prices, got {len(hourly_prices)}")

    # Create time points for hourly prices (0, 1, 2, ..., 23 hours)
    hourly_times = np.arange(24)

    # Create time points for 5-minute prices (0, 5/60, 10/60, ..., hours)
    # 288 intervals = 24 hours * 12 (5-min intervals per hour)
    five_min_times = np.arange(288) * (5.0 / 60.0)

    # Linear interpolation
    # For times beyond 23 hours, we extrapolate using the last value
    five_min_prices = np.interp(five_min_times, hourly_times, hourly_prices)

    return five_min_prices


def create_sample_prices(pattern: str = "peak") -> np.ndarray:
    """
    Create sample 24-hour price patterns for testing

    Args:
        pattern: Type of price pattern ("peak", "valley", "volatile", "flat")

    Returns:
        Array of 24 hourly prices ($/MWh)
    """
    if pattern == "peak":
        # High prices during day, low at night
        prices = np.array([
            30, 28, 26, 25, 27, 35,  # 00:00-05:00 - early morning
            45, 55, 60, 65, 70, 75,  # 06:00-11:00 - morning peak
            80, 78, 76, 80, 85, 90,  # 12:00-17:00 - afternoon peak
            85, 70, 55, 45, 38, 32   # 18:00-23:00 - evening decline
        ])
    elif pattern == "valley":
        # Low prices during day, high at night (unusual but possible)
        prices = np.array([
            60, 65, 70, 75, 70, 65,  # 00:00-05:00
            50, 40, 35, 30, 28, 26,  # 06:00-11:00
            25, 27, 30, 32, 35, 40,  # 12:00-17:00
            45, 50, 55, 60, 62, 61   # 18:00-23:00
        ])
    elif pattern == "volatile":
        # Highly volatile prices
        prices = np.array([
            30, 60, 25, 70, 35, 65,  # 00:00-05:00
            40, 80, 35, 75, 45, 85,  # 06:00-11:00
            50, 90, 45, 85, 55, 95,  # 12:00-17:00
            60, 80, 50, 70, 40, 60   # 18:00-23:00
        ])
    elif pattern == "flat":
        # Constant prices (no arbitrage opportunity)
        prices = np.full(24, 50.0)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return prices
