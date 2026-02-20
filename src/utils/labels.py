"""
标签生成模块

根据设计文档生成 SpikeEvent, LeadSpike, Regime 标签
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class SpikeLabels:
    """Spike 事件标签生成器"""

    @staticmethod
    def generate_spike_event(
        df: pd.DataFrame,
        zone: str = 'CPS',
        P_hi: float = 400,
        S_hi: float = 50,
        S_cross_hi: float = 80,
        m: int = 3,
        use_percentile: bool = False,
        percentile_window: int = 30
    ) -> pd.Series:
        """生成 SpikeEvent 标签

        Args:
            df: 数据框，包含价格和价差数据
            zone: 区域名称
            P_hi: 价格阈值 ($/MWh)
            S_hi: zone-hub 价差阈值 ($/MWh)
            S_cross_hi: zone-houston 价差阈值 ($/MWh)
            m: 持续时间阈值 (时间步数)
            use_percentile: 是否使用分位数阈值
            percentile_window: 滚动窗口天数（如果使用分位数）

        Returns:
            SpikeEvent_{zone}: 0/1 标签
        """
        # 条件 A: 价格高
        if use_percentile:
            window_size = percentile_window * 24 * 12  # 假设5分钟数据
            P_threshold = df[f'P_{zone}'].rolling(window=window_size, min_periods=1).quantile(0.99)
            cond_price = df[f'P_{zone}'] >= P_threshold
        else:
            cond_price = df[f'P_{zone}'] >= P_hi

        # 条件 B: 价差大 (约束主导)
        spread_zh = df[f'P_{zone}'] - df['P_Hub']

        if use_percentile:
            window_size = percentile_window * 24 * 12
            S_threshold = spread_zh.rolling(window=window_size, min_periods=1).quantile(0.95)
            cond_spread_zh = spread_zh >= S_threshold
        else:
            cond_spread_zh = spread_zh >= S_hi

        # 如果有 Houston 数据，添加跨区域价差条件
        if f'P_Houston' in df.columns and zone != 'Houston':
            spread_cross = df[f'P_{zone}'] - df['P_Houston']
            if use_percentile:
                S_cross_threshold = spread_cross.rolling(
                    window=window_size, min_periods=1
                ).quantile(0.95)
                cond_spread_cross = spread_cross >= S_cross_threshold
            else:
                cond_spread_cross = spread_cross >= S_cross_hi
            cond_spread = cond_spread_zh | cond_spread_cross
        else:
            cond_spread = cond_spread_zh

        # 条件 C: 持续时间
        raw_spike = cond_price & cond_spread
        sustained_spike = raw_spike.rolling(window=m, min_periods=1).sum() >= m

        return sustained_spike.fillna(False).astype(int)

    @staticmethod
    def generate_lead_spike(
        spike_event: pd.Series,
        H: int = 60,
        dt: int = 5
    ) -> pd.Series:
        """生成 LeadSpike 标签 (提前预警)

        Args:
            spike_event: SpikeEvent 标签序列
            H: 预警时间窗口 (分钟)
            dt: 数据时间分辨率 (分钟)

        Returns:
            LeadSpike: 0/1 标签
        """
        k = int(H / dt)  # 窗口大小

        # 反转 -> 滚动最大值 -> 再反转
        # 这样可以向前看未来窗口内是否有 spike
        lead_spike = spike_event[::-1].rolling(window=k, min_periods=1).max()[::-1]

        return lead_spike.fillna(0).astype(int)

    @staticmethod
    def generate_regime(
        df: pd.DataFrame,
        zone: str = 'CPS',
        P_mid: float = 150,
        S_mid: float = 20,
        spike_event: Optional[pd.Series] = None
    ) -> pd.Series:
        """生成 Regime 状态标签

        Args:
            df: 数据框
            zone: 区域名称
            P_mid: Tight 状态价格阈值
            S_mid: Tight 状态价差阈值
            spike_event: 已计算的 SpikeEvent（如果提供）

        Returns:
            Regime: 'Normal' / 'Tight' / 'Scarcity' 标签
        """
        spread_zh = df[f'P_{zone}'] - df['P_Hub']

        # 初始化为 Normal
        regime = pd.Series('Normal', index=df.index)

        # Tight 状态
        tight_cond = (df[f'P_{zone}'] >= P_mid) | (spread_zh >= S_mid)
        regime[tight_cond] = 'Tight'

        # Scarcity 状态 (优先级最高)
        if spike_event is not None:
            regime[spike_event == 1] = 'Scarcity'

        return regime


class LabelGenerator:
    """标签生成主类"""

    def __init__(
        self,
        zones: list = ['CPS', 'West', 'Houston'],
        P_hi: float = 400,
        S_hi: float = 50,
        S_cross_hi: float = 80,
        P_mid: float = 150,
        S_mid: float = 20,
        m: int = 3,
        H: int = 60,
        dt: int = 5
    ):
        """初始化

        Args:
            zones: 区域列表
            P_hi: Spike 价格阈值
            S_hi: Spike 价差阈值
            S_cross_hi: Spike 跨区域价差阈值
            P_mid: Tight 价格阈值
            S_mid: Tight 价差阈值
            m: Spike 持续时间阈值
            H: Lead Spike 预警窗口
            dt: 数据时间间隔
        """
        self.zones = zones
        self.P_hi = P_hi
        self.S_hi = S_hi
        self.S_cross_hi = S_cross_hi
        self.P_mid = P_mid
        self.S_mid = S_mid
        self.m = m
        self.H = H
        self.dt = dt

    def generate_all_labels(
        self,
        df: pd.DataFrame,
        use_percentile: bool = False
    ) -> pd.DataFrame:
        """生成所有标签

        Args:
            df: 原始数据
            use_percentile: 是否使用滚动分位数阈值

        Returns:
            包含所有标签的 DataFrame
        """
        labels = pd.DataFrame(index=df.index)

        for zone in self.zones:
            if f'P_{zone}' not in df.columns:
                print(f"警告: 缺少 P_{zone} 列，跳过 {zone} 区域")
                continue

            print(f"生成 {zone} 区域标签...")

            # 1. SpikeEvent
            spike_event = SpikeLabels.generate_spike_event(
                df,
                zone=zone,
                P_hi=self.P_hi,
                S_hi=self.S_hi,
                S_cross_hi=self.S_cross_hi,
                m=self.m,
                use_percentile=use_percentile
            )
            labels[f'SpikeEvent_{zone}'] = spike_event

            # 2. LeadSpike
            lead_spike = SpikeLabels.generate_lead_spike(
                spike_event,
                H=self.H,
                dt=self.dt
            )
            labels[f'LeadSpike_{zone}_{self.H}m'] = lead_spike

            # 3. Regime
            regime = SpikeLabels.generate_regime(
                df,
                zone=zone,
                P_mid=self.P_mid,
                S_mid=self.S_mid,
                spike_event=spike_event
            )
            labels[f'Regime_{zone}'] = regime

            # 统计信息
            n_spikes = spike_event.sum()
            n_lead_spikes = lead_spike.sum()
            regime_counts = regime.value_counts()

            print(f"  - SpikeEvent 数量: {n_spikes}")
            print(f"  - LeadSpike 数量: {n_lead_spikes}")
            print(f"  - Regime 分布:")
            for state, count in regime_counts.items():
                print(f"    {state}: {count} ({count/len(regime)*100:.2f}%)")

        return labels

    def identify_spike_events(
        self,
        spike_event: pd.Series,
        min_gap: int = 12  # 最小间隔（时间步数），用于分割独立事件
    ) -> list:
        """识别独立的 spike 事件

        Args:
            spike_event: SpikeEvent 标签序列
            min_gap: 最小间隔，小于此间隔的 spike 视为同一事件

        Returns:
            事件列表，每个事件为 {'start': timestamp, 'end': timestamp, 'duration': int, 'max_idx': timestamp}
        """
        events = []
        in_event = False
        event_start = None
        event_indices = []

        for idx, value in spike_event.items():
            if value == 1:
                if not in_event:
                    in_event = True
                    event_start = idx
                    event_indices = [idx]
                else:
                    event_indices.append(idx)
            else:
                if in_event:
                    # 检查是否应该结束事件
                    if len(event_indices) > 0:
                        gap = (idx - event_indices[-1]).total_seconds() / 60 / self.dt
                        if gap > min_gap:
                            # 记录事件
                            events.append({
                                'start': event_start,
                                'end': event_indices[-1],
                                'duration': len(event_indices),
                                'indices': event_indices.copy()
                            })
                            in_event = False
                            event_indices = []

        # 处理最后一个事件
        if in_event and len(event_indices) > 0:
            events.append({
                'start': event_start,
                'end': event_indices[-1],
                'duration': len(event_indices),
                'indices': event_indices.copy()
            })

        return events


if __name__ == '__main__':
    # 测试代码
    print("标签生成模块加载成功！")
    print("\n支持的标签类型:")
    print("1. SpikeEvent - Spike 事件标识")
    print("2. LeadSpike - 提前预警标签")
    print("3. Regime - 系统状态标签 (Normal/Tight/Scarcity)")
