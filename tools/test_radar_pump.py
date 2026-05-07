import time
from loguru import logger
from FlightController.Components.LDRadar_Driver import LD_Radar

def main():
    logger.info("=== 阶段 M3: 乐动 D500 雷达底层数据泵探活 ===")
    
    # 显式绑定你在 M1 阶段接管的物理串口
    RADAR_PORT = "/dev/ttySTM4" 
    
    # 初始化雷达驱动 (装载坐标系变换参数)
    radar = LD_Radar(name="D500_Test", index=0, mount_xy_cm=(0.0, 0.0), mount_yaw_deg=0.0)
    
    logger.info(f"正在接管 {RADAR_PORT} 并启动后台守护线程...")
    radar.start(com=RADAR_PORT, radar_type="D500")
    
    try:
        logger.info("预热等待 3 秒...")
        time.sleep(3)
        
        for i in range(10):
            if not radar.connected:
                logger.warning(f"[{i+1}/10] 雷达未连接，请检查 TX 引脚和 PWM 供电！")
            else:
                # 提取内存池中的状态参数
                rpm = radar.map.rotation_spd
                avail = radar.map.avail_points
                total = radar.map.total_points
                
                # 获取转换为直角坐标系的有效点云 (单位: cm)
                xy_points = radar.get_points_body_cm(max_distance_cm=200.0) # 只看2米内
                
                logger.success(f"[{i+1}/10] 链路通畅 | 转速: {rpm:.1f} RPM | 内存池饱和度: {avail}/{total} | 2米内有效点云数量: {len(xy_points)}")
                
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("人为中断测试。")
    finally:
        radar.stop()
        logger.info("雷达守护线程已安全销毁。")

if __name__ == "__main__":
    main()