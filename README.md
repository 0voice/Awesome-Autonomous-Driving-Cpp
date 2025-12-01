#  C++ 自动驾驶资源库
> **核心定位：** 这是一个专注于 **高性能 C++** 实现、**工程化落地** 和 **职业发展** 的自动驾驶资源精选集 🚗

## 📖 目录
- [🗺️ 学习路线图](#%EF%B8%8F-学习路线图)
- [✨ 核心算法讲解](#-核心算法讲解)
- [📚 学习资源（书籍/论文/课程）](#-学习资源书籍论文课程)
- [🛠️ 工具链](#-工具链)
- [🎓 面试八股文](#-面试八股文)
- [💼 招聘信息（2025最新）](#-招聘信息2025最新)
- [🤝 社区与贡献](#-社区与贡献)

## 🗺️ 学习路线图
<details>
<summary>点击展开</summary>
    
![Roadmap](./roadmap/roadmap.svg)

</details>

## ✨ 核心算法讲解
<details>
<summary>点击展开</summary>
    
- [数学与几何基础](core_content/README.md#数学与几何基础)
    - [Eigen](core_content/README.md#eigen)
    - [SO(3)、SE(3)、李代数](core_content/README.md#so3se3李代数)
    - [四元数与旋转表示](core_content/README.md#四元数与旋转表示)
    - [滤波器（KF/EKF/UKF/ESKF）](core_content/README.md#滤波器kfekfukfesef)

- [感知](core_content/README.md#感知)
    - [PointPillars](core_content/README.md#pointpillars)
    - [CenterPoint Voxel-to-BEV + CenterHead](core_content/README.md#centerpoint-voxel-to-bev--centerhead)
    - [多模态融合（激光雷达+相机）](core_content/README.md#多模态融合激光雷达相机)
    - [TensorRT 自定义插件开发](core_content/README.md#tensorrt-自定义插件开发)

- [定位](core_content/README.md#定位)
    - [NDT 配准](core_content/README.md#ndt-配准)
    - [FAST-LIO 紧耦合](core_content/README.md#fast-lio-紧耦合)
    - [ESKF 误差状态卡尔曼](core_content/README.md#eskf-误差状态卡尔曼)
    - [GPS/IMU 紧耦合](core_content/README.md#gpsimu-紧耦合)

- [建图](core_content/README.md#建图)
    - [离线建图](core_content/README.md#离线建图)
    - [在线回环检测](core_content/README.md#在线回环检测)
    - [高精地图与矢量地图](core_content/README.md#高精地图与矢量地图)

- [预测](core_content/README.md#预测)
    - [多目标跟踪](core_content/README.md#多目标跟踪)
    - [意图预测](core_content/README.md#意图预测)
    - [轨迹预测](core_content/README.md#轨迹预测)

- [规划](core_content/README.md#规划)
    - [Hybrid A* + Reeds-Shepp](core_content/README.md#hybrid-a--reeds-shepp)
    - [Lattice Planner](core_content/README.md#lattice-planner)
    - [EM Planner](core_content/README.md#em-planner)
    - [行为决策与状态机](core_content/README.md#行为决策与状态机)

- [控制](core_content/README.md#控制)
    - [MPC 横纵向解耦](core_content/README.md#mpc-横纵向解耦)
    - [LQR 与最优控制](core_content/README.md#lqr-与最优控制)
    - [Stanley / Pure Pursuit](core_content/README.md#stanley--pure-pursuit)
    - [车辆动力学模型](core_content/README.md#车辆动力学模型)

- [端到端](core_content/README.md#端到端)
    - [模仿学习](core_content/README.md#模仿学习)
    - [端到端模型 C++ 部署](core_content/README.md#端到端模型-c-部署)

- [仿真](core_content/README.md#仿真)
    - [CARLA C++ Client](core_content/README.md#carla-c-client)
    - [传感器仿真与同步](core_content/README.md#传感器仿真与同步)
    - [场景库与交通流](core_content/README.md#场景库与交通流)

- [中间件与通信](core_content/README.md#中间件与通信)
    - [Fast-DDS / CycloneDDS](core_content/README.md#fast-dds--cyclonedds)
    - [some/IP + vsomeip](core_content/README.md#someip--vsomeip)
    - [Protobuf 序列化](core_content/README.md#protobuf-序列化)
</details>
