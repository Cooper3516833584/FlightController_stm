from dataclasses import dataclass

import numpy as np


@dataclass
class PlannerConfig:
    max_speed_cm_s: float = 35.0
    approach_speed_cm_s: float = 20.0
    yaw_rate_limit_deg_s: float = 30.0
    obstacle_stop_distance_cm: float = 45.0
    obstacle_slow_distance_cm: float = 90.0
    target_center_deadband_px: float = 30.0
    camera_fov_deg: float = 70.0


@dataclass
class TargetObservation:
    center_px: tuple[float, float]
    image_size: tuple[int, int]
    confidence: float
    class_name: str = "target"


@dataclass
class VelocityCommand:
    vx_cm_s: float
    vy_cm_s: float
    vz_cm_s: float
    yaw_rate_deg_s: float
    reason: str


class LocalPlanner:
    def __init__(self, config: PlannerConfig | None = None):
        self.config = config or PlannerConfig()

    def plan(
        self,
        *,
        obstacles_body_cm: np.ndarray,
        target: TargetObservation | None,
    ) -> VelocityCommand:
        if target is None:
            return VelocityCommand(0.0, 0.0, 0.0, 0.0, "no_target")

        yaw_rate = self._target_yaw_rate(target)
        vx = self.config.approach_speed_cm_s
        reasons = ["target"]

        forward_distance = self._nearest_forward_obstacle_cm(obstacles_body_cm)
        if forward_distance is not None:
            if forward_distance < self.config.obstacle_stop_distance_cm:
                vx = 0.0
                reasons.append("obstacle_stop")
            elif forward_distance < self.config.obstacle_slow_distance_cm:
                vx *= 0.5
                reasons.append("obstacle_slow")

        vx = self._clip(vx, -self.config.max_speed_cm_s, self.config.max_speed_cm_s)
        yaw_rate = self._clip(
            yaw_rate,
            -self.config.yaw_rate_limit_deg_s,
            self.config.yaw_rate_limit_deg_s,
        )
        return VelocityCommand(vx, 0.0, 0.0, yaw_rate, "+".join(reasons))

    def _target_yaw_rate(self, target: TargetObservation) -> float:
        width, _ = target.image_size
        if width <= 0:
            return 0.0
        offset_px = target.center_px[0] - width / 2.0
        if abs(offset_px) <= self.config.target_center_deadband_px:
            return 0.0
        normalized_offset = offset_px / (width / 2.0)
        angle_error_deg = normalized_offset * (self.config.camera_fov_deg / 2.0)
        return self._clip(
            angle_error_deg,
            -self.config.yaw_rate_limit_deg_s,
            self.config.yaw_rate_limit_deg_s,
        )

    @staticmethod
    def _nearest_forward_obstacle_cm(obstacles_body_cm: np.ndarray) -> float | None:
        points = np.asarray(obstacles_body_cm, dtype=float)
        if points.size == 0:
            return None
        points = points.reshape(-1, 2)
        forward = points[(points[:, 0] > 0) & (np.abs(points[:, 1]) < 35.0)]
        if forward.size == 0:
            return None
        return float(np.min(forward[:, 0]))

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return float(np.clip(value, low, high))


__all__ = ["LocalPlanner", "PlannerConfig", "TargetObservation", "VelocityCommand"]
