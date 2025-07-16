from django.db import transaction
from .models import Flight_Trajectory

def generate_flight_trajectory(qar_instances):
    """
    根据 QAR 数据生成 Flight_Trajectory 记录，并计算变化量
    :param qar_instances: QAR 实例列表（已按 qar_id 和 dSimTime 排序）
    :return: List[Flight_Trajectory] 待插入的轨迹记录
    """
    trajectory_objects = []
    
    # 按 qar_id 分组，以便计算变化量
    qar_data_by_id = {}
    for qar in qar_instances:
        if qar.qar_id not in qar_data_by_id:
            qar_data_by_id[qar.qar_id] = []
        qar_data_by_id[qar.qar_id].append(qar)
    
    # 遍历每个 QAR ID，计算变化量
    for qar_id, qar_list in qar_data_by_id.items():
        # 按 dSimTime 排序，确保计算变化量的顺序正确
        qar_list_sorted = sorted(qar_list, key=lambda x: x.dSimTime)
        
        for i, qar in enumerate(qar_list_sorted):
            # 如果是第一条记录，变化量为 0
            if i == 0:
                longitude_change = qar.dLongitude
                latitude_change = qar.dLatitude
                asl_change = qar.dASL
                agl_change = qar.dAGL
            else:
                # 计算与前一条记录的差值
                prev_qar = qar_list_sorted[i - 1]
                longitude_change = qar.dLongitude - prev_qar.dLongitude
                latitude_change = qar.dLatitude - prev_qar.dLatitude
                asl_change = qar.dASL - prev_qar.dASL
                agl_change = qar.dAGL - prev_qar.dAGL
            
            # 创建 Flight_Trajectory 对象
            trajectory = Flight_Trajectory(
                qar_id=qar.qar_id,
                dSimTime=qar.dSimTime,
                dTrueHeading=qar.dTrueHeading,
                dMagHeading=qar.dMagHeading,
                dLongitude=qar.dLongitude,
                dLatitude=qar.dLatitude,
                dASL=qar.dASL,
                dAGL=qar.dAGL,
                dLongitude_change=longitude_change,
                dLatitude_change=latitude_change,
                dASL_change=asl_change,
                dAGL_change=agl_change,
            )
            trajectory_objects.append(trajectory)
    
    return trajectory_objects


def update_flight_trajectory(qar_instances):
    with transaction.atomic():
        # 1. 生成 Flight_Trajectory 记录
        trajectory_objects = generate_flight_trajectory(qar_instances)
        
        # 2. 批量插入 Flight_Trajectory
        Flight_Trajectory.objects.bulk_create(trajectory_objects, batch_size=1000)