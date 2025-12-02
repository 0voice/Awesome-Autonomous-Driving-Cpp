# Awesome C++ Autonomous Driving

[中文](https://github.com/0voice/Awesome-CPP-Autonomous-Driving/blob/main/README.md) | **English**

> **Core Positioning:** A curated collection focused on **high-performance C++**, **production-grade engineering**, and **career & interview preparation** in autonomous driving

## Table of Contents
- [🗺️ Learning Roadmap](#%EF%B8%8F-learning-roadmap)
- [✨ Core Topics Explained](#-core-topics-explained)
- [📚 Courses, Books & Papers](#-courses-books--papers)
- [📊 Datasets](#-datasets)
- [🛠️ Toolchain](#%EF%B8%8F-toolchain)
- [💻 Open-Source Projects](#-open-source-projects)
- [🎓 Interview Questions](#-interview-questions)
- [💼 Job Board](#-job-board)
- [🤝 Community & Contribution](#-community--contribution)

## 🗺️ Learning Roadmap

<details>
<summary>Click to expand</summary>
   
![Roadmap](./roadmap/roadmap.en.svg)

</details>

## ✨ Core Topics Explained


   
- [Math & Geometry](core_content/README.md#数学与几何基础)
    - [Eigen](core_content/README.md#eigen)
    - [SO(3), SE(3) & Lie Algebra](core_content/README.md#so3se3李代数)
    - [Quaternions & Rotation](core_content/README.md#四元数与旋转表示)
    - [Filters (KF/EKF/UKF/ESKF)](core_content/README.md#滤波器kfekfukfesef)
    - [Numerical Optimization (Ceres/g2o)](core_content/README.md#数值优化ceresg2o)
- [Perception](core_content/README.md#感知)
    - [PointPillars](core_content/README.md#pointpillars)
    - [CenterPoint Voxel-to-BEV + CenterHead](core_content/README.md#centerpoint-voxel-to-bev--centerhead)
    - [Multi-modal Fusion (LiDAR+Camera)](core_content/README.md#多模态融合激光雷达相机)
    - [TensorRT Custom Plugin Development](core_content/README.md#tensorrt-自定义插件开发)
- [Localization](core_content/README.md#定位)
    - [NDT Registration](core_content/README.md#ndt-配准)
    - [FAST-LIO Tightly-Coupled](core_content/README.md#fast-lio-紧耦合)
    - [ESKF Error-State Kalman](core_content/README.md#eskf-误差状态卡尔曼)
    - [GPS/IMU Tight Coupling](core_content/README.md#gpsimu-紧耦合)
- [Mapping](core_content/README.md#建图)
    - [Offline Mapping](core_content/README.md#离线建图)
    - [Online Loop Closure](core_content/README.md#在线回环检测)
    - [HD Maps & Vector Maps](core_content/README.md#高精地图与矢量地图)
- [Prediction](core_content/README.md#预测)
    - [Multi-Object Tracking](core_content/README.md#多目标跟踪)
    - [Intent Prediction](core_content/README.md#意图预测)
    - [Trajectory Prediction](core_content/README.md#轨迹预测)
- [Planning](core_content/README.md#规划)
    - [Hybrid A* + Reeds-Shepp](core_content/README.md#hybrid-a--reeds-shepp)
    - [Lattice Planner](core_content/README.md#lattice-planner)
    - [EM Planner](core_content/README.md#em-planner)
    - [Behavior Decision & State Machine](core_content/README.md#行为决策与状态机)
- [Control](core_content/README.md#控制)
    - [MPC Lateral-Longitudinal Decoupled](core_content/README.md#mpc-横纵向解耦)
    - [LQR & Optimal Control](core_content/README.md#lqr-与最优控制)
    - [Stanley / Pure Pursuit](core_content/README.md#stanley--pure-pursuit)
    - [Vehicle Dynamics Model](core_content/README.md#车辆动力学模型)
- [End-to-End](core_content/README.md#端到端)
    - [Imitation Learning](core_content/README.md#模仿学习)
    - [End-to-End Model C++ Deployment](core_content/README.md#端到端模型-c-部署)
- [Simulation](core_content/README.md#仿真)
    - [CARLA C++ Client](core_content/README.md#carla-c-client)
    - [Sensor Simulation & Synchronization](core_content/README.md#传感器仿真与同步)
    - [Scenario Library & Traffic Flow](core_content/README.md#场景库与交通流)
- [Middleware & Communication](core_content/README.md#中间件与通信)
    - [ROS/ROS2 Architecture](core_content/README.md#rosros-2-架构)
    - [Fast-DDS / CycloneDDS](core_content/README.md#fast-dds--cyclonedds)
    - [some/IP + vsomeip](core_content/README.md#someip--vsomeip)
    - [Protobuf Serialization](core_content/README.md#protobuf-序列化)


## 📚 Courses, Books & Papers



### Courses
| Course Name | Description |
|--------------------------------------------------------------------------|----------------------------------------------------------------------|
| [Self-Driving Cars Specialization](https://www.coursera.org/specializations/self-driving-cars) | The most complete 4-course autonomous driving series worldwide |
| [Introduction to Self-Driving Cars](https://www.coursera.org/learn/intro-self-driving-cars) | Best entry course with CARLA simulator project |
| [Motion Planning for Self-Driving Cars](https://www.coursera.org/learn/motion-planning-self-driving-cars) | Gold standard for planning: A*, Hybrid A*, Lattice, MPC all covered |
| [Visual Perception for Self-Driving Cars](https://www.coursera.org/learn/visual-perception-self-driving-cars) | Lane line, traffic light, 3D object detection full pipeline, OpenCV assignments easily ported to C++ |
| [State Estimation and Localization for Self-Driving Cars](https://www.coursera.org/learn/state-estimation-localization-self-driving-cars) | Kalman filter, particle filter, SLAM basics, matrix assignments naturally suitable for C++ |
| [Self-Driving Cars with Duckietown](https://www.edx.org/learn/technology/eth-zurich-self-driving-cars-with-duckietown) | ETH Zürich real-car course, full ROS2 + C++ practice, hardware-software combined |
| [Multi-Object Tracking for Automotive Systems](https://www.edx.org/learn/engineering/chalmers-university-of-technology-multi-object-tracking-for-automotive-systems) | Chalmers University multi-object tracking course, designed for automotive systems, includes SORT/Kalman fusion |
| [Autonomous Mobile Robots](https://www.edx.org/learn/autonomous-robotics/eth-zurich-autonomous-mobile-robots) | ETH Zürich mobile robot course, path planning/avoidance algorithms, applicable to autonomous driving localization |
| [Self-Driving Cars with Duckietown MOOC](https://duckietown.com/self-driving-cars-with-duckietown-mooc/) | ETH Zürich Duckietown hardware MOOC, AI robot autonomous decision-making, free hardware tutorial |
| [Advanced Kalman Filtering and Sensor Fusion](https://www.classcentral.com/course/udemy-advanced-kalman-filtering-and-sensor-fusion-401323) | Udemy advanced Kalman filter and sensor fusion, focused on autonomous driving C++ simulation |
| [Sensor Fusion Engineer Nanodegree](https://www.udacity.com/course/sensor-fusion-engineer--nd313) | Udacity sensor fusion program, LiDAR+Radar+Camera fusion, C++ real-time implementation |
| [Self-Driving Car Engineer Nanodegree](https://www.udacity.com/course/self-driving-car-engineer--nd013) | Udacity full-stack autonomous driving engineer, perception to planning, C++ deployment projects |
| [AI for Autonomous Vehicles and Robotics](https://www.coursera.org/learn/ai-for-autonomous-vehicles-and-robotics) | University of Michigan AI in autonomous vehicles, includes Kalman filter and decision-making |
| [The Complete Self-Driving Car Course - Applied Deep Learning](https://www.udemy.com/course/applied-deep-learningtm-the-complete-self-driving-car-course/) | Deep learning computer vision machine learning to build autonomous car, Python but easy to port to C++ |
| [Autonomous Aerospace Systems](https://www.coursera.org/learn/autonomous-aerospace-systems) | Autonomous aircraft software engineering, path planning/sensor fusion, applicable to autonomous driving |

### Books
| Book Title | Author | Description |
|------------------------------------------|-------------------------------------------|-----------------------------------|
| Introduction to Autonomous Vehicles (2nd Edition) | Rahul Kala | 1000+ pages full-stack autonomous driving textbook |
| Probabilistic Robotics | Sebastian Thrun / Wolfram Burgard / Dieter Fox | Standard textbook for probabilistic robotics, localization and SLAM |
| Planning Algorithms | Steven M. LaValle | Classic reference book in path planning field |
| Effective Modern C++ | Scott Meyers | Modern C++ best practices and code standards |
| C++ Concurrency in Action (2nd Edition) | Anthony Williams | C++ multi-threading and concurrent programming practice |
| C++ Templates: The Complete Guide (2nd Edition) | David Vandevoorde / Nicolai M. Josuttis / Douglas Gregor | Complete guide to C++ template metaprogramming |
| Multiple View Geometry in Computer Vision (2nd Edition) | Richard Hartley & Andrew Zisserman | Standard textbook for computer vision multi-view geometry |
| Vehicle Dynamics and Control (2nd Edition) | Rajesh Rajamani | Classic textbook for vehicle dynamics and control |
| Autonomous Driving: How the Driverless Revolution will Change the World | Andreas Herrmann et al. | Full panorama of autonomous driving industry + technical routes |
| Introduction to Autonomous Mobile Robots (2nd Edition) | Roland Siegwart et al. | Classic introduction to mobile robots, from sensors to navigation |
| State Estimation for Robotics | Timothy D. Barfoot | Modern derivation of Kalman filter, factor graph, iSAM |
| Principles of Robot Motion: Theory, Algorithms, and Implementations | Howie Choset et al. | Complete theoretical system of motion planning |
| Applied Predictive Control | Sunan Huang & Tan Kok Kiong | Most practical MPC textbook in autonomous driving |
| Model Predictive Control: Theory and Design | Rawlings & Mayne | Absolute standard textbook in MPC field, required reading for control group |
| Autonomous Vehicle Technology: A Guide for Policymakers and Planners | James M. Anderson et al. | Clear system architecture and module division, suitable for writing proposals |
| Learning OpenCV 4 (Vol.1 & Vol.2) | Adrian Kaehler & Gary Bradski | Official OpenCV book |
| Modern Robotics: Mechanics, Planning, and Control | Kevin M. Lynch & Frank C. Park | Modern textbook for robotic arms + mobile robots |
| The DARPA Urban Challenge | Martin Buehler et al. | Technical summary of 2007 DARPA champion teams, historical classic |
| [Deep Learning for Self-driving Car](https://www.princeton.edu/~alaink/Orf467F14/Deep%20Driving.pdf) | Chenyi Chen et al. (Princeton) | Classic deep learning end-to-end autonomous driving, includes C++ implementation ideas |
| [Self-Driving Vehicles and Enabling Technologies](https://www.intechopen.com/books/9869) | Marian Găiceanu et al. (ed.) | Full book chapters free PDF, includes C++ embedded chapter |
| [Autonomous Driving: Technical, Legal and Social Aspects](https://link.springer.com/content/pdf/10.1007/978-3-662-48847-8.pdf) | Markus Maurer et al. | Springer full book Open Access, technology + regulations + architecture |
| [Self-Driving Car Using Simulator](https://www.researchgate.net/publication/380180926_Self-Driving_Car_Using_Simulator/download) | ResearchGate project report | Complete C++ small car project + code, perfect for hands-on |
| [Self-Driving Cars: Are We Ready?](https://assets.kpmg.com/content/dam/kpmg/pdf/2013/10/self-driving-cars-are-we-ready.pdf) | KPMG | Classic industry report |
| [Self-Driving Car Autonomous System Overview](https://dadun.unav.edu/bitstream/10171/67589/1/2022.06.01%20TFG%20Daniel%20Casado%20Herraez.pdf) | Daniel Casado Herráez | Spanish university graduation project, C++ hardware interface real case |
| [Planning Algorithms](http://planning.cs.uiuc.edu/planning.pdf) | Steven M. LaValle | Absolute classic in path planning field, A*/RRT/PRM all included |
| [Probabilistic Robotics](https://www.probabilistic-robotics.org/Probabilistic_Robotics.pdf) | Sebastian Thrun et al. | Bible of probabilistic robotics, must-read for localization/SLAM |
| [Multiple View Geometry in Computer Vision (2nd Edition)](https://www.robots.ox.ac.uk/~vgg/hzbook/hzbook.pdf) | Hartley & Zisserman | God-tier book for multi-view geometry, essential for visual SLAM |
| [State Estimation for Robotics](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/AF9E1F4F7D0D7B8F6D8B8E8F9E0F1A2B/9781107159396ar.pdf/State_Estimation_for_Robotics.pdf) | Timothy D. Barfoot | Clearest modern Kalman/factor graph textbook |

### Papers
| Title | Author | Year | Venue | Description |
|-------|--------|------|-------|-------------|
| [End-to-End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316.pdf) | Mariusz Bojarski et al. (NVIDIA) | 2016 | arXiv | End-to-end pioneering work, C++ real-time deployment classic |
| [CARLA: An Open Urban Driving Simulator](https://arxiv.org/pdf/1711.03938.pdf) | Alexey Dosovitskiy et al. | 2017 | CoRL | Official CARLA paper, must-read C++ Client API |
| [Learning by Cheating](https://arxiv.org/pdf/1912.12294.pdf) | Dian Chen, Vladlen Koltun | 2019 | CoRL | Imitation learning + planning ceiling, extremely high C++ engineering reference value |
| [Planning-oriented Autonomous Driving](https://arxiv.org/pdf/2212.10156.pdf) | Yihan Hu et al. | 2023 | CVPR | Currently hottest planning-oriented framework, official C++ code open-sourced |
| [TransFuser](https://arxiv.org/pdf/2205.15997.pdf) | Kashyap Chitta et al. | 2022 | CVPR | Transformer multi-sensor fusion SOTA, mainstream C++ deployment |
| [NEAT: Neural Attention Fields](https://arxiv.org/pdf/2309.04442.pdf) | Kaustubh Mani et al. | 2023 | ICCV | 2023 latest end-to-end, lightweight C++ inference extremely fast |
| [ST-P3](https://arxiv.org/pdf/2207.07601.pdf) | Shengchao Hu et al. | 2022 | ECCV | Spatio-temporal Transformer, one of the best C++ real-time performance |
| [Efficient Lidar Odometry for Autonomous Driving](https://arxiv.org/pdf/2209.06828.pdf) | Junha Kim et al. (KAIST) | 2022 | RA-L | Pure LiDAR odometry, lightweight C++ implementation, embedded-friendly |
| [VISTA 2.0](https://arxiv.org/pdf/2211.00931.pdf) | Alexander Amini et al. | 2022 | IROS | Data-driven simulator, C++ multi-modal sensor simulation |
| [DriveAdapter](https://arxiv.org/pdf/2309.01243.pdf) | Xiaosong Jia et al. | 2023 | ICCV | Perception-planning decoupled new paradigm, best C++ modular practice |
| [OpenOccupancy: A Large Scale Benchmark](https://arxiv.org/pdf/2303.03991.pdf) | Xiaofeng Wang, Zheng Zhu et al. | 2023 | ICCV | Largest open-source Occupancy dataset, official C++ benchmark code |


## 📊 Datasets


  
- [KITTI](https://www.cvlibs.net/datasets/kitti/raw_data.php)  
  Classic 3D perception benchmark for 3D object detection, tracking, and odometry
- [nuScenes](https://www.nuscenes.org/download)  
  Large-scale multi-modal dataset focusing on full-scene 3D detection and trajectory prediction
- [Waymo Open Dataset](https://waymo.com/open/download)  
  Industry-leading finely annotated dataset, ideal for high-precision perception and LiDAR processing
- [Argoverse 2](https://www.argoverse.org/av2.html)  
  Comes with HD vector maps, focused on trajectory prediction, map fusion, and driving behavior analysis
- [A2D2 (Audi)](https://www.a2d2.audi/en/download/)  
  Includes CAN bus data, used for semantic segmentation and multi-modal 3D annotation
- [comma2k19](https://github.com/commaai/comma2k19)  
  Monocular camera + real driving CAN data, best suited for end-to-end driving models
- [CARLA Generated Data](https://carla.readthedocs.io/en/latest/download/)  
  Open-source simulator, customizable weather/maps, generates perfectly synchronized multi-sensor data infinitely
- [ApolloScape](https://apolloscape.auto/)  
  Street view images, LiDAR point clouds, trajectory data covering all aspects of urban traffic perception and navigation
- [Cityscapes](https://www.cityscapes-dataset.com/)  
  Urban street video sequences with fine pixel-level semantic and instance segmentation annotations
- [SemanticKITTI](https://www.semantic-kitti.org/)  
  KITTI extension with semantic segmentation labels for LiDAR point clouds, focused on 3D scene understanding
- [WoodScape](https://woodscape.valeo.com/)  
  Fisheye camera images for surround-view semantic segmentation, suitable for parking and low-speed scenarios
- [Zenseact Open Dataset (ZOD)](https://zod.zenseact.com/)  
  Multi-modal European urban driving data including frame sequences, driving logs, and radar point clouds
- [NVIDIA Physical AI Autonomous Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)  
  Multi-sensor global driving data covering 25+ countries and 2500+ cities, focused on end-to-end physical AI
- [MAN TruckScenes](https://brandportal.man/d/QSf8mPdU5Hgj)  
  Multi-modal truck driving dataset covering diverse conditions such as bad weather and multi-lane roads
- [Para-Lane](https://nizqleo.github.io/paralane-dataset/)  
  Real-world multi-lane dataset designed for novel view synthesis and end-to-end driving evaluation
- [UniOcc](https://huggingface.co/datasets/tasl-lab/uniocc)  
  Occupancy grid prediction and voxel flow dataset, supporting cross-domain generalization and future occupancy prediction
- [InterHub](https://www.nature.com/articles/s41597-025-05344-7)  
  Dense multi-agent interaction trajectory data from large-scale naturalistic driving records, focused on driving interaction research
- [rounD](https://arxiv.org/html/2401.01454v1)  
  Roundabout road user trajectory dataset with 6 hours of video and 13K+ user records, supporting behavior prediction
- [WOMD-Reasoning](https://waymo.com/open/download)  
  Language-annotated dataset based on Waymo Open Motion Dataset, focused on interaction intent description and reasoning
- [V2V-QA](https://eddyhkchiu.github.io/v2vllm.github.io/)  
  Vehicle-to-vehicle question-answering dataset, supporting LLM methods for end-to-end cooperative autonomous driving
- [DriveBench](https://drive-bench.github.io/)  
  Vision-language model reliability benchmark dataset with 19K frames and 20K QA pairs, covering various driving tasks
- [FutureSightDrive](https://github.com/MIV-XJTU/FSDrive)  
  Spatio-temporal chain-of-thought dataset, supporting vision-driven autonomous driving prediction and planning
- [Adverse Weather Dataset](https://light.princeton.edu/datasets/automated_driving_dataset/)  
  Adverse weather multi-modal dataset with 12K real samples and 1.5K controlled samples under snow/rain/fog conditions



## 🛠️ Toolchain


  
- [Apollo](https://github.com/ApolloAuto/apollo)  
  Baidu's complete open-source L4 autonomous driving platform covering perception, planning, control, and simulation
- [Autoware](https://autoware.org/)  
  World's largest open-source autonomous driving software stack based on ROS 2, covering full urban road scenarios
- [OpenPilot](https://github.com/commaai/openpilot)  
  comma.ai open-source end-to-end driving system, already running on tens of thousands of real vehicles
- [ROS 2](https://docs.ros.org/en/rolling/Installation.html)  
  Most widely used middleware in robotics and autonomous driving, supporting distributed real-time systems
- [CyberRT](https://github.com/ApolloAuto/apollo/tree/master/cyber)  
  Apollo's self-developed high-performance data communication and scheduling framework
- [CARLA](https://carla.org/)  
  High-fidelity autonomous driving simulator based on Unreal Engine, supporting multi-sensor and traffic flow
- [LGSVL Simulator / SVL](https://www.svlsimulator.com/)  
  Former LG open-source simulator, perfect support for Apollo/Autoware closed-loop testing
- [NVIDIA DRIVE Sim](https://developer.nvidia.com/drive/drive-sim)  
  NVIDIA enterprise-grade autonomous driving simulation platform based on Omniverse
- [DeepStream SDK](https://developer.nvidia.com/deepstream-sdk)  
  NVIDIA intelligent video analysis and multi-sensor fusion pipeline framework
- [TensorRT](https://developer.nvidia.com/tensorrt)  
  NVIDIA high-performance deep learning inference engine optimized for embedded and in-vehicle use
- [ONNX Runtime](https://onnxruntime.ai/)  
  Microsoft open-source cross-platform inference engine supporting multiple hardware acceleration
- [Triton Inference Server](https://github.com/triton-inference-server/server)  
  NVIDIA open-source high-concurrency model deployment and inference service framework
- [Bazel](https://bazel.build/)  
  Google's large-scale build and test tool, Apollo's default build system
- [Colcon](https://colcon.readthedocs.io/)  
  ROS 2 official recommended meta-build tool
- [Fast-DDS](https://www.eprosima.com/)  
  eProsima high-performance DDS implementation, default communication middleware for ROS 2
- [Cyclone DDS](https://cyclonedds.io/)  
  Eclipse Foundation high-performance DDS implementation, widely used in automotive and robotics
- [Zenoh](https://zenoh.io/)  
  Next-generation ultra-low-latency edge communication protocol, validated by multiple autonomous driving companies
- [Foxglove Studio](https://foxglove.dev/)  
  Most popular data visualization and analysis tool for autonomous driving and robotics
- [Mcap](https://mcap.dev/)  
  Next-generation cross-platform recording file format, replacing rosbag
- [Lanelet2](https://github.com/fzi-forschungszentrum-informatik/Lanelet2)  
  Open-source HD map format and routing library, Autoware's default map solution
- [AUTOSAR Adaptive](https://www.autosar.org/standards/adaptive-platform/)  
  Next-generation in-vehicle adaptive software platform standard supporting dynamic updates and service-oriented architecture


## 💻 Open-Source Projects


  
- [Apollo](https://github.com/ApolloAuto/apollo)  
  Baidu L4 full-stack autonomous driving platform covering perception, planning, control, and simulation, supporting real vehicle deployment
- [Autoware](https://github.com/autowarefoundation/autoware)  
  World's largest ROS2-based open-source autonomous driving system, already running on real vehicles in multiple countries
- [openpilot](https://github.com/commaai/openpilot)  
  comma.ai end-to-end driving system, real L2++ benchmark running on 200,000+ vehicles
- [CARLA](https://github.com/carla-simulator/carla)  
  High-fidelity autonomous driving simulator based on Unreal Engine, supporting multi-sensor and traffic flow
- [LGSVL Simulator](https://github.com/lgsvl/simulator)  
  Former LG simulator, perfect closed-loop testing support for Apollo/Autoware
- [UniAD](https://github.com/OpenDriveLab/UniAD)  
  Open-source full end-to-end autonomous driving pipeline (perception-prediction-planning-control)
- [DriveDreamer-2](https://github.com/f1yfisher/DriveDreamer2)  
  Open-source end-to-end driving model that can generate world models and drive directly in CARLA
- [OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy)  
  Official Occupancy/4D-Occupancy open-source implementation
- [VAD](https://github.com/hustvl/VAD)  
  End-to-end autonomous driving open-source (vectorized trajectory output)
- [ST-P3](https://github.com/OpenDriveLab/ST-P3)  
  Transformer one-stop end-to-end autonomous driving, fully connecting perception-prediction-planning
- [Donkey Car](https://github.com/autorope/donkeycar)  
  Real 1:10 scale car hardware + complete deep learning project
- [AirSim](https://github.com/microsoft/AirSim)  
  Microsoft open-source simulator supporting autonomous driving and drone multi-platform simulation
- [Webots](https://github.com/cyberbotics/webots)  
  Open-source robot simulator supporting ROS2 and autonomous vehicle physics simulation


## 🎓 Interview Questions



### Perception Engineer
- Describe common point cloud filtering methods (voxel, statistical, pass-through) and their applicable scenarios
- Design a point cloud ground segmentation algorithm to solve complex terrain problems
- How to improve point cloud object detection accuracy? Describe feature extraction and classifier design process
- Implement KD-Tree based nearest neighbor search to calculate k nearest points for a given point
- Implement lane line detection algorithm (using OpenCV perspective transform and Hough transform)
- How to solve camera image quality degradation in rain/fog? Design multi-modal fusion scheme
- In YOLO model, how to design loss function to improve small object detection accuracy?
- Explain BEVDet perception algorithm principle, including feature extraction, BEV conversion and detection head design
- Design camera and LiDAR extrinsic calibration scheme, including calibration board selection and optimization method
- In multi-sensor fusion system, how to handle time synchronization problem? Compare hardware and software synchronization schemes
- Implement Kalman filter based sensor data fusion, fusing millimeter-wave radar and camera target position information

### Decision & Planning Engineer
- Compare Dijkstra, A*, RRT* algorithms advantages and disadvantages in autonomous driving path planning and applicable scenarios
- Design highway automatic lane change decision algorithm, considering front/rear vehicle distance, speed difference and safety gap
- Implement a trajectory planner to generate smooth vehicle trajectory (continuous curvature) satisfying vehicle dynamics constraints
- Design unprotected left turn decision logic, considering oncoming traffic, pedestrians, traffic signals and road rules
- In urban roads, how to handle "ghost probe" (suddenly appearing pedestrian) situation? Design emergency decision mechanism
- Implement reinforcement learning based decision system to solve complex intersection traffic problem (reward function design, state representation)
- Design vehicle intent prediction model, predict other vehicles driving intention based on historical trajectory and surrounding environment
- How to integrate traffic rules (right of way, speed limit) into decision system? Design rule engine
- In multi-agent scenarios, how to handle other vehicles not following traffic rules? Design robust decision strategy

### Control Engineer
- Establish vehicle two-degree-of-freedom dynamics model (bicycle model), derive state equation and control input
- In vehicle steering control, how to handle "understeer" and "oversteer" problems? Design compensation strategy
- Derive relationship between vehicle sideslip angle, yaw rate and steering wheel angle
- Design LQR-based vehicle lateral controller (lane keeping), including state selection, weight matrix design and discretization implementation
- Implement model predictive control (MPC) to solve vehicle longitudinal control (car following) problem, considering actuator delay and road slope
- How to adjust PID controller parameters to adapt to different speeds and road conditions? Design adaptive PID strategy
- Design automatic parking control system to implement parallel and perpendicular parking functions, considering parking space detection and trajectory planning
- In high-speed driving, how to handle front wheel blowout and other emergency situations? Design emergency control strategy
- Implement vehicle stability control (ESC) to prevent sideslip and tail swing, design control algorithm based on tire force observation

### System Development (C++ Direction)
- Describe differences and usage scenarios of C++ four smart pointers (shared_ptr/unique_ptr/weak_ptr/auto_ptr)
- Implement thread-safe singleton pattern (C++11+), considering double-checked locking and static local variable schemes
- Explain RAII (Resource Acquisition Is Initialization) principle and how to apply it in autonomous driving system?
- How does memory alignment affect point cloud processing performance? Take Eigen library matrix as example
- Design efficient obstacle trajectory data structure to support real-time query and update
- Implement a memory pool to manage frequently allocated and released small objects in autonomous driving system
- Design software architecture of autonomous driving perception system, considering multi-threading, data pipeline and error handling
- How to design modular system to achieve low coupling and high cohesion between perception, decision and control modules?
- When deploying deep learning models on embedded platforms (such as Jetson AGX), what optimizations are needed?

### Embedded Software Engineer
- Explain the difference between RTOS (real-time operating system) and general operating system, why is it important in autonomous driving?
- Design multi-level interrupt system to ensure real-time performance of critical tasks (such as braking control), using Cortex-M NVIC priority grouping
- In multi-core RTOS, how to implement inter-task communication and synchronization? Compare mailbox, semaphore and message queue schemes
- Describe CAN bus frame structure, compare differences and applicable scenarios between standard frame and extended frame
- Implement CAN bus communication protocol, including ID allocation, arbitration mechanism and error handling
- Design vehicle system diagnosis scheme based on UDS (Unified Diagnostic Services), implement fault code reading and clearing
- Write GPIO control program to implement vehicle lights, wipers and other peripheral control
- Design ADC sampling program to read vehicle sensor (such as tire pressure, oil temperature) data, considering anti-interference and precision optimization
- In embedded systems, how to handle power management? Design low-power mode and wake-up mechanism

### SLAM & Localization Engineer
- Describe ORB-SLAM2/3 system workflow, including feature extraction, tracking, local mapping and loop closure detection
- How to solve scale drift problem in visual SLAM? Compare monocular, stereo and RGB-D schemes
- In dynamic scenes, how to detect and remove moving objects? Design method based on optical flow and semantic segmentation
- Implement key point cloud processing part of LOAM or Lego-LOAM algorithm, including feature extraction and matching
- In LiDAR SLAM, how to handle point cloud distortion caused by vehicle motion? Design motion compensation scheme
- Compare advantages and disadvantages of LiDAR-SLAM and visual-SLAM, how to fuse them in autonomous driving?
- Design visual+IMU+RTK+LiDAR fusion localization system, including time synchronization and extrinsic calibration
- Implement EKF/UKF-based multi-sensor fusion localization, fusing GPS, IMU and wheel speedometer data
- In urban canyon and tunnel where GPS signal is lost, how to ensure localization accuracy? Design auxiliary localization scheme

### HD Map Engineer
- Design high-precision map construction process based on LiDAR point cloud, including point cloud registration, feature extraction and map element generation
- How to evaluate HD map quality? Design evaluation metrics for accuracy, completeness and consistency
- In map construction, how to handle dynamic obstacles (such as moving vehicles)? Design dynamic object filtering and completion scheme
- Design efficient lane-level map data structure to support fast query and update
- How to implement incremental update of HD map? Design difference detection and transmission scheme to reduce bandwidth consumption
- On embedded devices, how to optimize map storage and retrieval? Design hierarchical indexing and caching mechanism
- Describe application scenarios of HD map in autonomous driving, such as localization, path planning and decision-making
- How to encode traffic rules (no left turn, speed limit) into HD map? Design map semantic representation
- In autonomous driving system, how to achieve fast matching (localization) between map and vehicle position? Design efficient search algorithm

### Testing Engineer
- Design test case library for L4 autonomous driving system, covering perception, decision and control functions
- How to test autonomous driving system performance in extreme weather (heavy rain, dense fog, ice and snow)? Design test scenarios
- Implement scenario-based testing to test autonomous driving system decision logic
- Compare V (verification) and V (validation) processes in autonomous driving testing, explain their respective purposes and methods
- Design safety testing scheme for autonomous driving system to verify safety degradation mechanism in failure situations
- In HIL (hardware-in-the-loop) testing, how to simulate sensors and actuators? Design test platform
- Design risk matrix for autonomous driving testing, identify high-risk scenarios and formulate testing strategies
- In real vehicle testing, how to collect and analyze data to optimize algorithms? Design data collection and analysis process
- For the "long-tail problem" (rare but dangerous scenarios) of autonomous driving system, how to design test cases?

### Model Deployment & Optimization Engineer
- Design model quantization scheme (FP32→FP16→INT8) to improve inference speed while maintaining accuracy
- Implement model pruning to remove redundant parameters and reduce model size, design pruning criteria and fine-tuning strategy
- In model distillation, how to design teacher model and student model? How to choose distillation loss function?
- For autonomous driving perception model, design model parallelism and data parallelism scheme to improve multi-GPU inference efficiency
- Implement ONNX model to TensorRT conversion and optimization, configure appropriate workspace and precision mode
- On embedded platform, how to optimize model inference performance? Compare GPU, NPU and CPU schemes
- Design autonomous driving model service architecture to support high concurrency and low latency inference
- In end-to-end system, how to optimize data preprocessing and post-processing pipeline to reduce overall latency?
- Implement model hot update, update model without restarting service to ensure service continuity

### General Questions
- What is the core difference between process and thread? How to choose in actual development?
- What are the inter-process communication (IPC) methods? Their advantages and disadvantages and applicable scenarios?
- What is the working principle of virtual memory? Why do we need virtual memory?
- What is deadlock? What are the four necessary conditions for deadlock? How to avoid deadlock?
- What is the difference between user mode and kernel mode? How to switch?
- What is the difference and correspondence between OSI seven-layer model and TCP/IP four-layer model?
- What is the process of TCP three-way handshake? Why do we need three-way handshake?
- How does TCP ensure reliable data transmission? What are the mechanisms?
- What is the difference between HTTP and HTTPS? What is the working principle of HTTPS?
- Complete process from entering URL to page display?
- What is the core idea of Von Neumann architecture?
- What is the composition and working principle of CPU?
- What is the working principle of Cache? Cache hit rate and failure types?
- What is the role of memory alignment? How does it affect program performance?
- What is the principle and role of DMA technology? Why can it improve IO performance?
- What is the difference and applicable scenarios between TCP and UDP?
- What are the common page replacement algorithms? Why is LRU commonly used?
- What are common HTTP status codes and their meanings?
- What are critical resources and critical sections? How to solve critical section problem?
- What are the classification and role of bus? Differences between data bus, address bus and control bus?

### C++
- [C++ High-frequency Interview Questions](https://github.com/0voice/cpp-learning-2025/blob/main/interview_questions/README.en.md)



## 💼 Job Board



- **Waymo** – [Apply Here](https://waymo.com/careers/)
- **Cruise** – [Apply Here](https://getcruise.com/careers/)
- **Zoox** – [Apply Here](https://zoox.com/careers)
- **Aurora** – [Apply Here](https://aurora.tech/careers/)
- **NVIDIA** – [Apply Here](https://nvidia.com/en-us/about-nvidia/careers/) 
- **Motional** – [Apply Here](https://motional.com/careers)
- **Applied Intuition** – [Apply Here](https://www.appliedintuition.com/careers)
- **Waabi** – [Apply Here](https://waabi.ai/careers/)
- **Oxa** – [Apply Here](https://oxa.tech/careers/)


## 🤝 Community & Contribution



Thank you for visiting!  
This repo aims to be the strongest C++ autonomous driving resource collection worldwide.

Contributions of any kind are extremely welcome — new projects, fixes, translations, interview questions, etc.

Star & Watch so you never miss an update!

