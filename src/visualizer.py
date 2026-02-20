"""
Visualization utilities for battery optimization results
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Optional

from .optimizer import OptimizationResult
from .battery_config import BatteryConfig


class BatteryVisualizer:
    """
    Visualizer for battery optimization results
    """

    def __init__(self, config: BatteryConfig):
        """
        Initialize visualizer

        Args:
            config: Battery configuration
        """
        self.config = config

    def plot_optimization_results(
        self,
        result: OptimizationResult,
        prices: np.ndarray,
        save_path: Optional[str] = None,
        figsize: tuple = (15, 10)
    ):
        """
        Create comprehensive visualization of optimization results

        Args:
            result: Optimization result
            prices: Price array used for optimization
            save_path: Path to save figure (optional)
            figsize: Figure size (width, height)
        """
        T = len(prices)
        time_hours = np.arange(T) * self.config.delta_t

        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

        # Plot 1: Prices
        self._plot_prices(axes[0], time_hours, prices)

        # Plot 2: Power (charging/discharging)
        self._plot_power(axes[1], time_hours, result)

        # Plot 3: State of Charge
        self._plot_soc(axes[2], time_hours, result)

        # Plot 4: Cumulative Revenue
        self._plot_cumulative_revenue(axes[3], time_hours, result, prices)

        # Add title and metadata
        fig.suptitle(
            f"Battery Energy Storage Optimization Results\n"
            f"Total Revenue: ${result.total_revenue:.2f} | "
            f"Status: {result.status} | "
            f"Solve Time: {result.solve_time:.3f}s",
            fontsize=14,
            fontweight="bold"
        )

        plt.xlabel("Time (hours)")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to {save_path}")

        plt.show()

    def _plot_prices(self, ax, time_hours, prices):
        """Plot electricity prices"""
        ax.plot(time_hours, prices, color="blue", linewidth=2, label="LMP Price")
        ax.fill_between(time_hours, prices, alpha=0.3, color="blue")
        ax.set_ylabel("Price ($/MWh)", fontsize=11)
        ax.set_title("Electricity Prices", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_power(self, ax, time_hours, result: OptimizationResult):
        """Plot charging and discharging power"""
        ax.fill_between(
            time_hours,
            0,
            -result.P_ch,
            color="red",
            alpha=0.6,
            label="Charging"
        )
        ax.fill_between(
            time_hours,
            0,
            result.P_dis,
            color="green",
            alpha=0.6,
            label="Discharging"
        )
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
        ax.axhline(
            y=self.config.P_dis_max,
            color="green",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            label="Max Discharge"
        )
        ax.axhline(
            y=-self.config.P_ch_max,
            color="red",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            label="Max Charge"
        )
        ax.set_ylabel("Power (MW)", fontsize=11)
        ax.set_title("Charging/Discharging Power", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_soc(self, ax, time_hours, result: OptimizationResult):
        """Plot state of charge"""
        ax.plot(
            time_hours,
            result.SoC * 100,
            color="purple",
            linewidth=2.5,
            label="SoC"
        )
        ax.axhline(
            y=self.config.SoC_min * 100,
            color="red",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            label="Min SoC"
        )
        ax.axhline(
            y=self.config.SoC_max * 100,
            color="red",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            label="Max SoC"
        )
        ax.fill_between(
            time_hours,
            self.config.SoC_min * 100,
            result.SoC * 100,
            alpha=0.3,
            color="purple"
        )
        ax.set_ylabel("State of Charge (%)", fontsize=11)
        ax.set_title("Battery State of Charge", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_cumulative_revenue(
        self,
        ax,
        time_hours,
        result: OptimizationResult,
        prices
    ):
        """Plot cumulative revenue over time"""
        # Calculate instantaneous revenue at each time step (match optimizer objective)
        instant_revenue = (
            prices * result.P_dis * self.config.delta_t
            - prices * result.P_ch * self.config.delta_t
            - self.config.c_deg * result.P_dis * self.config.delta_t
            - self.config.lambda_delta_p * (result.ramp_up + result.ramp_down)
        )
        cumulative_revenue = np.cumsum(instant_revenue)

        ax.plot(
            time_hours,
            cumulative_revenue,
            color="darkgreen",
            linewidth=2.5,
            label="Cumulative Revenue"
        )
        ax.fill_between(
            time_hours,
            0,
            cumulative_revenue,
            alpha=0.3,
            color="darkgreen"
        )
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
        ax.set_ylabel("Revenue ($)", fontsize=11)
        ax.set_title("Cumulative Revenue", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

    def plot_power_vs_price(
        self,
        result: OptimizationResult,
        prices: np.ndarray,
        save_path: Optional[str] = None,
        figsize: tuple = (12, 6)
    ):
        """
        Create scatter plot showing relationship between price and power decisions

        Args:
            result: Optimization result
            prices: Price array
            save_path: Path to save figure (optional)
            figsize: Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Charging vs Price
        charging_mask = result.P_ch > 0.01  # Small threshold to avoid zeros
        if np.any(charging_mask):
            ax1.scatter(
                prices[charging_mask],
                result.P_ch[charging_mask],
                alpha=0.6,
                color="red",
                s=50
            )
        ax1.set_xlabel("Price ($/MWh)", fontsize=11)
        ax1.set_ylabel("Charging Power (MW)", fontsize=11)
        ax1.set_title("Charging Power vs Price", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # Discharging vs Price
        discharging_mask = result.P_dis > 0.01
        if np.any(discharging_mask):
            ax2.scatter(
                prices[discharging_mask],
                result.P_dis[discharging_mask],
                alpha=0.6,
                color="green",
                s=50
            )
        ax2.set_xlabel("Price ($/MWh)", fontsize=11)
        ax2.set_ylabel("Discharging Power (MW)", fontsize=11)
        ax2.set_title("Discharging Power vs Price", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def print_summary(self, result: OptimizationResult, prices: np.ndarray):
        """
        Print text summary of optimization results

        Args:
            result: Optimization result
            prices: Price array
        """
        print("=" * 70)
        print("BATTERY ENERGY STORAGE OPTIMIZATION SUMMARY")
        print("=" * 70)
        print(f"\nOptimization Status: {result.status}")
        print(f"Solve Time: {result.solve_time:.3f} seconds")
        print(f"\nTotal Revenue: ${result.total_revenue:.2f}")

        # Energy statistics
        total_charged = np.sum(result.P_ch) * self.config.delta_t
        total_discharged = np.sum(result.P_dis) * self.config.delta_t
        print(f"\nEnergy Charged: {total_charged:.2f} MWh")
        print(f"Energy Discharged: {total_discharged:.2f} MWh")
        print(f"Energy Throughput: {total_charged + total_discharged:.2f} MWh")

        # Round-trip efficiency
        if total_charged > 0:
            effective_efficiency = total_discharged / total_charged
            print(f"Effective Round-Trip Efficiency: {effective_efficiency:.2%}")

        # Price statistics
        avg_charge_price = (
            np.sum(prices * result.P_ch) / np.sum(result.P_ch)
            if np.sum(result.P_ch) > 0 else 0
        )
        avg_discharge_price = (
            np.sum(prices * result.P_dis) / np.sum(result.P_dis)
            if np.sum(result.P_dis) > 0 else 0
        )

        print(f"\nAverage Charging Price: ${avg_charge_price:.2f}/MWh")
        print(f"Average Discharging Price: ${avg_discharge_price:.2f}/MWh")
        print(f"Price Spread Captured: ${avg_discharge_price - avg_charge_price:.2f}/MWh")

        # SoC statistics
        print(f"\nInitial SoC: {result.SoC[0]:.1%}")
        print(f"Final SoC: {result.SoC[-1]:.1%}")
        print(f"Min SoC: {np.min(result.SoC):.1%}")
        print(f"Max SoC: {np.max(result.SoC):.1%}")

        # Cycle count (rough estimate)
        soc_range = np.max(result.SoC) - np.min(result.SoC)
        cycles = soc_range / 2  # Rough approximation
        print(f"\nApproximate Cycles: {cycles:.2f}")

        print("=" * 70)
