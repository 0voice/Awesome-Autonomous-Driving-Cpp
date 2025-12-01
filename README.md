#  C++ 自动驾驶资源库
> **核心定位：** 这是一个专注于 **高性能 C++** 实现、**工程化落地** 和 **职业发展** 的自动驾驶资源精选集 🚗

## 📖 目录
- [🗺️ 学习路线图](#%EF%B8%8F-学习路线图)
- [✨ 核心内容讲解](#-核心内容讲解)
- [📚 学习资源（课程/书籍/论文）](#-学习资源)
- [📊 数据集](#-数据集)
- [🛠️ 工具链](#-工具链)
- [🎓 面试八股文](#-面试八股文)
- [💼 招聘信息（2025最新）](#-招聘信息2025最新)
- [🤝 社区与贡献](#-社区与贡献)

## 🗺️ 学习路线图
<details>
<summary>点击展开</summary>
    
![Roadmap](./roadmap/roadmap.svg)

</details>

## ✨ 核心内容讲解
<details>
<summary>点击展开</summary>
    
- [数学与几何基础](core_content/README.md#数学与几何基础)
    - [Eigen](core_content/README.md#eigen)
    - [SO(3)、SE(3)、李代数](core_content/README.md#so3se3李代数)
    - [四元数与旋转表示](core_content/README.md#四元数与旋转表示)
    - [滤波器（KF/EKF/UKF/ESKF）](core_content/README.md#滤波器kfekfukfesef)
    - [数值优化 (Ceres/g2o)](core_content/README.md#数值优化ceresg2o)

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
    - [ROS/ROS 2 架构](core_content/README.md#rosros-2-架构)
    - [Fast-DDS / CycloneDDS](core_content/README.md#fast-dds--cyclonedds)
    - [some/IP + vsomeip](core_content/README.md#someip--vsomeip)
    - [Protobuf 序列化](core_content/README.md#protobuf-序列化)
</details>

## 📚 学习资源
<details>
<summary>点击展开</summary>

### 课程

| 课程名称                                                                 | 简介                                                                 |
|--------------------------------------------------------------------------|----------------------------------------------------------------------|
| [Self-Driving Cars Specialization](https://www.coursera.org/specializations/self-driving-cars) | 全球最完整的自动驾驶四门套课，感知+定位+规划+控制全栈                |
| [Introduction to Self-Driving Cars](https://www.coursera.org/learn/intro-self-driving-cars) | 自动驾驶入门第一课，含CARLA模拟器项目                                |
| [Motion Planning for Self-Driving Cars](https://www.coursera.org/learn/motion-planning-self-driving-cars) | 规划方向天花板，A*、Hybrid A*、Lattice、MPC全覆盖                    |
| [Visual Perception for Self-Driving Cars](https://www.coursera.org/learn/visual-perception-self-driving-cars) | 车道线、信号灯、3D目标检测完整流程，OpenCV作业极易转C++              |
| [State Estimation and Localization for Self-Driving Cars](https://www.coursera.org/learn/state-estimation-localization-self-driving-cars) | 卡尔曼滤波、粒子滤波、SLAM入门，矩阵作业天然适合C++实现             |
| [Self-Driving Cars with Duckietown](https://www.edx.org/learn/technology/eth-zurich-self-driving-cars-with-duckietown) | 苏黎世联邦理工实车小车课，ROS2 + C++全程实战，软硬件结合             |
| [Multi-Object Tracking for Automotive Systems](https://www.edx.org/learn/engineering/chalmers-university-of-technology-multi-object-tracking-for-automotive-systems) | 查尔姆斯大学多目标跟踪课，专为汽车系统设计，含SORT/Kalman融合        |
| [Autonomous Mobile Robots](https://www.edx.org/learn/autonomous-robotics/eth-zurich-autonomous-mobile-robots) | ETH Zurich移动机器人课，路径规划/避障算法，适用于自驾车定位          |
| [Self-Driving Cars with Duckietown MOOC](https://duckietown.com/self-driving-cars-with-duckietown-mooc/) | ETH Zurich Duckietown硬件MOOC，AI机器人自主决策，免费硬件教程        |
| [Advanced Kalman Filtering and Sensor Fusion](https://www.classcentral.com/course/udemy-advanced-kalman-filtering-and-sensor-fusion-401323) | Udemy高级卡尔曼滤波和传感器融合，专注自驾车C++模拟实现               |
| [Sensor Fusion Engineer Nanodegree](https://www.udacity.com/course/sensor-fusion-engineer--nd313) | Udacity传感器融合专项，LiDAR+Radar+Camera融合，C++实时实现           |
| [Self-Driving Car Engineer Nanodegree](https://www.udacity.com/course/self-driving-car-engineer--nd013) | Udacity自驾车工程师全栈，感知到规划，C++部署项目                     |
| [AI for Autonomous Vehicles and Robotics](https://www.coursera.org/learn/ai-for-autonomous-vehicles-and-robotics) | 密歇根大学AI在自驾车中的应用，含Kalman滤波和决策，边缘计算扩展       |
| [The Complete Self-Driving Car Course - Applied Deep Learning](https://www.udemy.com/course/applied-deep-learningtm-the-complete-self-driving-car-course/) | 深度学习计算机视觉机器学习构建自主车，Python但易转C++                |
| [Autonomous Aerospace Systems](https://www.coursera.org/learn/autonomous-aerospace-systems) | 自驾飞行器软件工程，路径规划/传感器融合，适用于自驾车                |

    
### 书籍
| 书籍名称                                      | 作者                                      | 简介                              |
|------------------------------------------|-------------------------------------------|-----------------------------------|
| 无人驾驶车辆系统概论（第2版）            | Rahul Kala                               | 1000+页自动驾驶全栈教材 |
| 自动驾驶技术系列·决策与规划              | 清华大学智能产业研究院（AIR团队）        | 国内最全面的规划算法书 |
| 无人驾驶原理与实践                        | 刘少山等（兰州大学）                      | 完整C++工程代码，从零搭建L4小车   |
| Probabilistic Robotics                   | Sebastian Thrun / Wolfram Burgard / Dieter Fox | 概率机器人学标准教材，定位与SLAM |
| Planning Algorithms                      | Steven M. LaValle                        | 路径规划领域经典参考书            |
| Effective Modern C++                     | Scott Meyers                             | 现代C++最佳实践与代码规范         |
| C++ Concurrency in Action（第2版）       | Anthony Williams                         | C++多线程与并发编程实战           |
| C++ Templates: The Complete Guide（第2版）| David Vandevoorde / Nicolai M. Josuttis / Douglas Gregor | C++模板元编程完整指南 |
| Multiple View Geometry in Computer Vision（第2版） | Richard Hartley & Andrew Zisserman | 计算机视觉多视图几何标准教材     |
| Vehicle Dynamics and Control（第2版）    | Rajesh Rajamani                          | 车辆动力学与控制经典教材          |
| Autonomous Driving: How the Driverless Revolution will Change the World | Andreas Herrmann 等              | 自动驾驶产业全景+技术路线，适合开阔眼界 |
| Introduction to Autonomous Mobile Robots（第2版） | Roland Siegwart 等               | 移动机器人入门经典，传感器到导航全讲   |
| State Estimation for Robotics                  | Timothy D. Barfoot               | 卡尔曼滤波、因子图、iSAM现代推导 |
| Principles of Robot Motion: Theory, Algorithms, and Implementations | Howie Choset 等              | 运动规划完整理论体系   |
| Applied Predictive Control                     | Sunan Huang & Tan Kok Kiong      | 自动驾驶里最实用的MPC教材   |
| Model Predictive Control: Theory and Design    | Rawlings & Mayne                 | MPC领域绝对标准教材，控制组必备        |
| Autonomous Vehicle Technology: A Guide for Policymakers and Planners | James M. Anderson 等        | 系统架构与模块划分清晰，适合写方案   |
| Learning OpenCV 4（Vol.1 & Vol.2）             | Adrian Kaehler & Gary Bradski    | OpenCV官方书         |
| Modern Robotics: Mechanics, Planning, and Control | Kevin M. Lynch & Frank C. Park | 机械臂+移动机器人现代教材 |
| The DARPA Urban Challenge                      | Martin Buehler 等                | 2007年DARPA冠军队伍技术总结，历史经典  |
| [Deep Learning for Self-driving Car](https://www.princeton.edu/~alaink/Orf467F14/Deep%20Driving.pdf)           | Chenyi Chen 等 (Princeton)        | 深度学习端到端自动驾驶经典，含C++实现思路         |
| [Self-Driving Vehicles and Enabling Technologies](https://www.intechopen.com/books/9869)       | Marian Găiceanu 等 (编)           | 全书章节免费PDF，含C++嵌入式章节              |
| [Autonomous Driving: Technical, Legal and Social Aspects](https://link.springer.com/content/pdf/10.1007/978-3-662-48847-8.pdf) | Markus Maurer 等           | Springer整书Open Access，技术+法规+架构     |
| [Self-Driving Car Using Simulator](https://www.researchgate.net/publication/380180926_Self-Driving_Car_Using_Simulator/download) | ResearchGate 项目报告     | 完整C++小车项目+代码，适合直接上手            |
| [Self-Driving Cars: Are We Ready?](https://assets.kpmg.com/content/dam/kpmg/pdf/2013/10/self-driving-cars-are-we-ready.pdf) | KPMG                       | 经典产业报告        |
| [Self-Driving Car Autonomous System Overview](https://dadun.unav.edu/bitstream/10171/67589/1/2022.06.01%20TFG%20Daniel%20Casado%20Herraez.pdf) | Daniel Casado Herráez     | 西班牙大学生毕业设计，C++硬件接口实战案例     |
| [Planning Algorithms](http://planning.cs.uiuc.edu/planning.pdf)                    | Steven M. LaValle                 | 路径规划领域绝对经典，A*/RRT/PRM全有          |
| [Probabilistic Robotics](https://www.probabilistic-robotics.org/Probabilistic_Robotics.pdf) | Sebastian Thrun 等             | 概率机器人学圣经，定位/SLAM必读               |
| [Multiple View Geometry in Computer Vision（第2版）](https://www.robots.ox.ac.uk/~vgg/hzbook/hzbook.pdf) | Hartley & Zisserman          | 多视图几何神书，视觉SLAM必备                  |
| [State Estimation for Robotics](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/AF9E1F4F7D0D7B8F6D8B8E8F9E0F1A2B/9781107159396ar.pdf/State_Estimation_for_Robotics.pdf) | Timothy D. Barfoot | 现代卡尔曼/因子图最清晰教材                   | 

### 论文
| 论文标题                                                                                             | 作者                                           | 年份 | 会议/期刊 | 简介                              |
|--------------------------------------------------------------------------------------------------------------------|------------------------------------------------|------|-----------|-----------------------------------------------------|
| [End-to-End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316.pdf)                                  | Mariusz Bojarski et al. (NVIDIA)              | 2016 | arXiv     | 端到端开山之作，C++实时部署经典案例                 |
| [CARLA: An Open Urban Driving Simulator](https://arxiv.org/pdf/1711.03938.pdf)                                     | Alexey Dosovitskiy et al.                     | 2017 | CoRL      | CARLA官方论文，C++ Client API必读                   |
| [Learning by Cheating](https://arxiv.org/pdf/1912.12294.pdf)                                                       | Dian Chen, Vladlen Koltun                     | 2019 | CoRL      | 模仿学习+规划天花板，C++工程实现参考价值极高         |
| [Planning-oriented Autonomous Driving](https://arxiv.org/pdf/2212.10156.pdf)                                       | Yihan Hu et al.                               | 2023 | CVPR      | 目前最火的规划导向框架，官方C++代码已开源           |
| [TransFuser](https://arxiv.org/pdf/2205.15997.pdf)                                                                 | Kashyap Chitta et al.                         | 2022 | CVPR      | Transformer多传感器融合SOTA，C++部署主流            |
| [NEAT: Neural Attention Fields](https://arxiv.org/pdf/2309.04442.pdf)                                             | Kaustubh Mani et al.                          | 2023 | ICCV      | 2023最新端到端，轻量C++推理极快                     |
| [ST-P3](https://arxiv.org/pdf/2207.07601.pdf)                                                                       | Shengchao Hu et al.                           | 2022 | ECCV      | 时空Transformer，C++实时性最强之一                  |
| [Efficient Lidar Odometry for Autonomous Driving](https://arxiv.org/pdf/2209.06828.pdf)                           | Junha Kim et al. (KAIST)                      | 2022 | RA-L      | 纯激光里程计，轻量C++实现，嵌入式友好               |
| [VISTA 2.0](https://arxiv.org/pdf/2211.00931.pdf)                                                                  | Alexander Amini et al.                        | 2022 | IROS      | 数据驱动仿真器，C++多模态传感器模拟                 |
| [DriveAdapter](https://arxiv.org/pdf/2309.01243.pdf)                                                              | Xiaosong Jia et al.                           | 2023 | ICCV      | 感知-规划解耦新范式，C++模块化最佳实践              |
| [OpenOccupancy: A Large Scale Benchmark](https://arxiv.org/pdf/2303.03991.pdf)                                     | Xiaofeng Wang, Zheng Zhu et al.               | 2023 | ICCV      | 最大开源Occupancy数据集，官方C++基准代码            |


</details>
