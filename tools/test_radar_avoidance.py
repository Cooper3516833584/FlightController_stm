"""
单雷达在线避障链路地面测试。

验证闭环: D500雷达 → 点云 → 障碍物检测 → 飞控速度指令。

用法:
    # 仅测试雷达端（不连飞控，不发指令）
    python tools/test_radar_avoidance.py --no-fc --dry-run

    # 雷达 + 飞控，但不发送实际指令
    python tools/test_radar_avoidance.py --dry-run

    # 完整链路（雷达 + 飞控 + 发送指令）
    python tools/test_radar_avoidance.py
"""

import argparse
import sys
import time
from pathlib import Path


def _setup_path() -> None:
    root = Path(__file__).resolve().parents[1]
    for p in (root, root.parent):
        value = str(p)
        if value not in sys.path:
            sys.path.insert(0, value)


def main() -> None:
    _setup_path()

    from FlightController import FC_Controller
    from FlightController.Components.LDRadar_Driver import LD_Radar
    from FlightController.Solutions.LocalPlanner import LocalPlanner, PlannerConfig, VelocityCommand
    from loguru import logger

    parser = argparse.ArgumentParser(
        description="单雷达在线避障链路地面测试",
    )
    parser.add_argument(
        "--port",
        default="/dev/ttySTM4",
        help="雷达串口路径 (默认: /dev/ttySTM4)",
    )
    parser.add_argument(
        "--fc-port",
        default=None,
        help="飞控串口路径 (默认: 自动探测)",
    )
    parser.add_argument(
        "--no-fc",
        action="store_true",
        help="不连接飞控，仅测试雷达端",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="不发送实际飞控指令（但仍连接飞控读取状态）",
    )
    parser.add_argument(
        "--max-distance-cm",
        type=float,
        default=300.0,
        help="障碍物检测最大距离/cm (默认: 300)",
    )
    parser.add_argument(
        "--stop-distance-cm",
        type=float,
        default=80.0,
        help="急停距离/cm (默认: 80)",
    )
    parser.add_argument(
        "--slow-distance-cm",
        type=float,
        default=150.0,
        help="减速距离/cm (默认: 150)",
    )
    parser.add_argument(
        "--cruise-speed-cm-s",
        type=float,
        default=30.0,
        help="巡航速度/cm/s (默认: 30)",
    )
    parser.add_argument(
        "--corridor-half-width-cm",
        type=float,
        default=50.0,
        help="前方走廊半宽/cm (默认: 50)",
    )
    parser.add_argument(
        "--loop-hz",
        type=float,
        default=10.0,
        help="主循环频率/Hz (默认: 10)",
    )
    args = parser.parse_args()

    # ---------- 初始化雷达 ----------
    logger.info(f"正在连接雷达 {args.port} ...")
    radar = LD_Radar(
        name="Avoidance_Test",
        index=0,
        mount_xy_cm=(0.0, 0.0),
        mount_yaw_deg=0.0,
    )
    try:
        radar.start(com=args.port, radar_type="D500")
    except (RuntimeError, OSError) as e:
        logger.error(f"雷达串口启动失败: {e}")
        return

    # ---------- 初始化飞控 ----------
    fc = None
    if not args.no_fc:
        logger.info("正在连接飞控...")
        fc = FC_Controller()
        try:
            fc.start_listen_serial(serial_dev=args.fc_port, block_until_connected=True)
            fc.wait_for_connection()
            state = fc.state
            logger.info(
                f"FC 已连接 | mode={state.mode.value} unlock={state.unlock.value} "
                f"bat={state.bat.value:.1f}V alt={state.alt_add.value}cm"
            )
        except Exception as e:
            logger.error(f"飞控连接失败: {e}")
            fc = None

    # ---------- 初始化规划器 ----------
    planner_config = PlannerConfig(
        enable_free_flight=True,
        free_flight_speed_cm_s=args.cruise_speed_cm_s,
        max_speed_cm_s=50.0,
        obstacle_stop_distance_cm=args.stop_distance_cm,
        obstacle_slow_distance_cm=args.slow_distance_cm,
        forward_corridor_half_width_cm=args.corridor_half_width_cm,
    )
    planner = LocalPlanner(config=planner_config)

    logger.info(
        f"规划器参数: stop<{args.stop_distance_cm}cm "
        f"slow<{args.slow_distance_cm}cm "
        f"cruise={args.cruise_speed_cm_s}cm/s "
        f"corridor=±{args.corridor_half_width_cm}cm"
    )

    # ---------- 等待雷达预热 ----------
    logger.info("等待雷达预热 3 秒...")
    time.sleep(3)

    if not radar.connected:
        logger.error("雷达未连接！请检查 TX 引脚和 PWM 供电。")
        radar.stop()
        if fc is not None:
            fc.close()
        return

    # ---------- 主循环 ----------
    period = 1.0 / max(args.loop_hz, 0.1)
    logger.info(f"开始避障主循环 @ {args.loop_hz}Hz (周期 {period:.3f}s)")
    logger.info("按 Ctrl+C 停止...")

    loop_count = 0
    try:
        while True:
            t_start = time.perf_counter()

            # 1. 检查雷达连接状态
            if not radar.connected:
                logger.warning("雷达断连！")
                if fc is not None and not args.dry_run:
                    fc.send_realtime_control_data(0, 0, 0, 0)
                time.sleep(0.5)
                continue

            # 2. 获取机体坐标系下的障碍点云
            obstacles = radar.get_points_body_cm(
                max_distance_cm=args.max_distance_cm
            )

            # 3. 计算前方最近障碍物距离
            forward_dist = planner._nearest_forward_obstacle_cm(obstacles)

            # 4. 规划避障决策
            command = planner.plan(obstacles_body_cm=obstacles, target=None)

            # 5. 发送到飞控
            if fc is not None and not args.dry_run:
                fc.send_realtime_control_data(
                    round(command.vx_cm_s),
                    round(command.vy_cm_s),
                    round(command.vz_cm_s),
                    round(command.yaw_rate_deg_s),
                )

            # 6. 日志输出 (每10次循环打印一次详细信息，其余打印摘要)
            if loop_count % 10 == 0:
                fc_state_str = ""
                if fc is not None:
                    try:
                        s = fc.state
                        fc_state_str = (
                            f"FC[mode={s.mode.value} unlock={s.unlock.value} "
                            f"bat={s.bat.value:.1f}V alt={s.alt_add.value}cm]"
                        )
                    except Exception:
                        fc_state_str = "FC[state_read_error]"

                logger.info(
                    f"[#{loop_count:04d}] {fc_state_str} | "
                    f"点云={len(obstacles)}点 | "
                    f"前方={forward_dist:.0f}cm" if forward_dist is not None else "[#{loop_count:04d}] 前方=无 | "
                    f"指令=(vx={command.vx_cm_s:.0f}, vy={command.vy_cm_s:.0f}, "
                    f"vz={command.vz_cm_s:.0f}, yaw={command.yaw_rate_deg_s:.0f}) | "
                    f"原因={command.reason}"
                )
            else:
                dist_str = f"{forward_dist:.0f}cm" if forward_dist is not None else "---"
                logger.debug(
                    f"[#{loop_count:04d}] 前方={dist_str} "
                    f"vx={command.vx_cm_s:.0f} reason={command.reason}"
                )

            loop_count += 1

            # 7. 控制循环频率
            elapsed = time.perf_counter() - t_start
            remaining = period - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        logger.info("收到中断信号，正在安全退出...")
    finally:
        # 发送零速度
        if fc is not None and not args.dry_run:
            logger.info("发送零速度指令...")
            fc.send_realtime_control_data(0, 0, 0, 0)
            time.sleep(0.1)

        radar.stop()
        if fc is not None:
            fc.close()

        logger.info(f"测试结束 | 共运行 {loop_count} 个循环")


if __name__ == "__main__":
    main()
