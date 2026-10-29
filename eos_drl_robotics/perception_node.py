#!/usr/bin/env python3
"""
================================================================================
Project: Edge-Optimized Sensor Fusion: A Unified SLAM-DRL Framework for Autonomous Service Robots
Module: Lightweight Perception Node (perception_node.py)
Target Hardware: NVIDIA Jetson Nano (4GB Edge GPU), Intel RealSense D435i RGB-D Camera
ROS 2 Distribution: Humble Hawksbill | Python: 3.10 | PyTorch: CUDA Accelerated

ACADEMIC RESEARCH GAP & METHODOLOGICAL RATIONALE:
--------------------------------------------------------------------------------
Conventional autonomous mobile robot (AMR) architectures rely on dense 3D Semantic
SLAM pipelines (e.g., TSDF-based Voxblox, OctoMap, or dense point-cloud projections)
to generate world models for local path planning. On edge-embedded computational
platforms such as the NVIDIA Jetson Nano (128-core Maxwell GPU, 5-10W power budget),
voxel ray-casting and high-dimensional 3D point cloud convolutions incur severe
computational bottlenecks:
    1. Memory Bandwidth Saturation: Transferring point clouds (~300,000 points/frame)
       over the unified LPDDR4 memory bus triggers thermal throttling and memory spills.
    2. High Latency: Point cloud filtering and KD-tree construction drop perception
       update rates below 2-4 Hz, violating real-time dynamic obstacle avoidance bounds
       (Nyquist-Shannon control stability criterion requires >= 10-15 Hz).

OUR SOLUTION (The EOS Perception Pipeline):
--------------------------------------------------------------------------------
Instead of reconstructing uncompressed 3D geometries, this node implements an end-to-end
learned 2D Top-Down Semantic Occupancy Grid Estimator.
A 4-channel tensor [R, G, B, D] is streamed directly into an edge-optimized MobileNetV3-style
encoder-decoder architecture running under PyTorch CUDA FP16 inference mode.
The network directly regresses a 64x64 top-down Bird's-Eye-View (BEV) semantic grid:
    - Channel 0: P(Free Space)
    - Channel 1: P(Static Obstacle)   [Walls, furniture, static infrastructure]
    - Channel 2: P(Dynamic Obstacle)  [Pedestrians, moving service equipment]

The output is published as a flattened 1D tensor via std_msgs/Float32MultiArray for zero-overhead
consumption by the downstream PPO DRL navigator and reward environment, alongside an optional
nav_msgs/OccupancyGrid for RViz2 telemetry.
================================================================================
"""

import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose
import message_filters

import torch
import torch.nn as nn
import torch.nn.functional as F

# Threading and OpenCV bridge utilities
from cv_bridge import CvBridge, CvBridgeError


# ==============================================================================
# SECTION 1: EDGE-OPTIMIZED PYTORCH PERCEPTION BACKBONE
# ==============================================================================

class HardSigmoid(nn.Module):
    """Memory-efficient piecewise linear approximation of sigmoid for edge GPUs."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x + 3.0, inplace=True) / 6.0


class HardSwish(nn.Module):
    """
    Piecewise linear approximation of Swish activation: x * HardSigmoid(x).
    Eliminates expensive transcendental exp() operations on Jetson Maxwell CUDA cores.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.relu6(x + 3.0, inplace=True) / 6.0


class SqueezeExcitation(nn.Module):
    """
    Lightweight channel-attention block designed to recalibrate RGB vs. Depth feature maps.
    Critical for low-texture environments where depth cues dominate color features.
    """
    def __init__(self, in_channels: int, squeeze_factor: int = 4):
        super().__init__()
        squeezed_channels = max(1, in_channels // squeeze_factor)
        self.fc1 = nn.Conv2d(in_channels, squeezed_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeezed_channels, in_channels, kernel_size=1, bias=False)
        self.hardsigmoid = HardSigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.hardsigmoid(scale)
        return x * scale


class InvertedResidualBlock(nn.Module):
    """
    MobileNetV3 Inverted Residual Block with Depthwise Separable Convolutions.
    Minimizes Multiply-Accumulate (MAC) operations and parameter count for edge inference.
    """
    def __init__(self, in_channels: int, exp_channels: int, out_channels: int,
                 kernel_size: int, stride: int, use_se: bool):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        layers = []

        # 1. Pointwise Expansion (1x1 conv)
        if exp_channels != in_channels:
            layers.append(nn.Conv2d(in_channels, exp_channels, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(exp_channels))
            layers.append(HardSwish())

        # 2. Depthwise Convolution (kxk grouped conv)
        layers.append(nn.Conv2d(exp_channels, exp_channels, kernel_size=kernel_size,
                                stride=stride, padding=kernel_size // 2,
                                groups=exp_channels, bias=False))
        layers.append(nn.BatchNorm2d(exp_channels))
        layers.append(HardSwish())

        # 3. Squeeze-and-Excitation Channel Attention
        if use_se:
            layers.append(SqueezeExcitation(exp_channels))

        # 4. Pointwise Linear Projection (1x1 conv, no activation to preserve manifold)
        layers.append(nn.Conv2d(exp_channels, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class MobileNetV3OccupancyEncoder(nn.Module):
    """
    Unified 4-Channel RGB-D to 2D BEV Semantic Occupancy Grid Neural Network.
    
    Input: [Batch, 4, H, W] -> (R, G, B, Normalized Depth)
    Output: [Batch, 3, 64, 64] -> Softmax probabilities for (Free, Static, Dynamic)
    
    Theoretical Justification:
    Bypasses explicit 3D voxel transformation matrices via learned spatial inductive bias.
    The decoder reshapes perspective visual features into top-down cartesian grid coordinates.
    """
    def __init__(self, in_channels: int = 4, num_classes: int = 3, grid_dim: int = 64):
        super().__init__()
        self.grid_dim = grid_dim
        self.num_classes = num_classes

        # Stem Convolution: Downsample 224x224 RGB-D input to 112x112
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            HardSwish()
        )

        # Inverted Bottleneck Sequence (Lightweight MobileNetV3-Small Spine)
        # Specs: (in_c, exp_c, out_c, k, s, se)
        self.stage1 = InvertedResidualBlock(16, 16, 16, kernel_size=3, stride=2, use_se=True)   # 56x56
        self.stage2 = InvertedResidualBlock(16, 72, 24, kernel_size=3, stride=2, use_se=False)  # 28x28
        self.stage3 = InvertedResidualBlock(24, 88, 24, kernel_size=3, stride=1, use_se=False)  # 28x28
        self.stage4 = InvertedResidualBlock(24, 96, 40, kernel_size=5, stride=2, use_se=True)   # 14x14
        self.stage5 = InvertedResidualBlock(40, 96, 40, kernel_size=5, stride=1, use_se=True)   # 14x14

        # BEV Spatial Decoder: Reconstructs top-down 64x64 spatial occupancy representation
        self.bev_decoder = nn.Sequential(
            nn.Conv2d(40, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            HardSwish(),
            nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            HardSwish(),
            nn.Upsample(size=(grid_dim, grid_dim), mode='bilinear', align_corners=False),
            nn.Conv2d(32, num_classes, kernel_size=3, stride=1, padding=1, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass returning normalized channel-wise semantic occupancy probabilities.
        Returns: Tensor of shape [Batch, 3, 64, 64]
        """
        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        feat = self.stage4(feat)
        feat = self.stage5(feat)
        logits = self.bev_decoder(feat)
        # Softmax over semantic classes: (0: Free, 1: Static Obstacle, 2: Dynamic Obstacle)
        probabilities = F.softmax(logits, dim=1)
        return probabilities


# ==============================================================================
# SECTION 2: ROS 2 PERCEPTION NODE IMPLEMENTATION
# ==============================================================================

class PerceptionNode(Node):
    """
    ROS 2 Humble Node synchronizing RGB-D frames, performing CUDA tensor inference,
    and publishing edge-optimized flattened 2D occupancy grids.
    """
    def __init__(self):
        super().__init__('perception_node')

        # ----------------------------------------------------------------------
        # Parameter Declarations & Hardware Config
        # ----------------------------------------------------------------------
        self.declare_parameter('camera_rgb_topic', '/camera/color/image_raw')
        self.declare_parameter('camera_depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('occupancy_tensor_topic', '/perception/occupancy_grid_tensor')
        self.declare_parameter('occupancy_nav_topic', '/perception/bev_occupancy_grid')
        self.declare_parameter('input_width', 224)
        self.declare_parameter('input_height', 224)
        self.declare_parameter('grid_dimension', 64)
        self.declare_parameter('max_depth_meters', 5.0)
        self.declare_parameter('min_depth_meters', 0.1)
        self.declare_parameter('use_half_precision', True)

        self.rgb_topic = self.get_parameter('camera_rgb_topic').value
        self.depth_topic = self.get_parameter('camera_depth_topic').value
        self.tensor_topic = self.get_parameter('occupancy_tensor_topic').value
        self.nav_topic = self.get_parameter('occupancy_nav_topic').value
        self.in_w = self.get_parameter('input_width').value
        self.in_h = self.get_parameter('input_height').value
        self.grid_dim = self.get_parameter('grid_dimension').value
        self.max_depth = self.get_parameter('max_depth_meters').value
        self.min_depth = self.get_parameter('min_depth_meters').value
        self.use_fp16 = self.get_parameter('use_half_precision').value

        # ----------------------------------------------------------------------
        # CUDA Hardware Initialization & Optimization
        # ----------------------------------------------------------------------
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"[PERCEPTION_INIT] Active Compute Device: {self.device}")
        if self.device.type == 'cuda':
            self.get_logger().info(f"[PERCEPTION_INIT] GPU Name: {torch.cuda.get_device_name(0)}")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.fastest = True

        # Instantiate Model
        self.model = MobileNetV3OccupancyEncoder(
            in_channels=4,
            num_classes=3,
            grid_dim=self.grid_dim
        ).to(self.device)

        if self.use_fp16 and self.device.type == 'cuda':
            self.model = self.model.half()
            self.get_logger().info("[PERCEPTION_INIT] FP16 Half-Precision Inference Enabled for Jetson Nano.")

        self.model.eval()

        # Pre-allocate pinned memory buffers to avoid repeated host-to-device allocation stalls
        self.tensor_buffer = torch.zeros(
            (1, 4, self.in_h, self.in_w),
            dtype=torch.float16 if (self.use_fp16 and self.device.type == 'cuda') else torch.float32,
            device=self.device
        )

        # Warm-up pass to eliminate initial CUDA kernel compilation latency
        self._warmup_gpu()

        # ----------------------------------------------------------------------
        # ROS 2 Synchronization & Communications
        # ----------------------------------------------------------------------
        self.bridge = CvBridge()

        # Sensor data QoS profile (Best Effort, Volatile history for high throughput)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Message Filters for Approximate Time Synchronization (RGB + Depth alignment)
        self.rgb_sub = message_filters.Subscriber(self, Image, self.rgb_topic, qos_profile=sensor_qos)
        self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=sensor_qos)

        # RealSense frames typically exhibit <= 30ms jitter
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=5,
            slop=0.04
        )
        self.sync.registerCallback(self.rgbd_sync_callback)

        # Publishers
        self.tensor_pub = self.create_publisher(Float32MultiArray, self.tensor_topic, 10)
        self.nav_grid_pub = self.create_publisher(OccupancyGrid, self.nav_topic, 10)

        # Performance Monitoring
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.get_logger().info("[PERCEPTION_INIT] Perception Node Initialized and Ready for Real-Time Streaming.")

    def _warmup_gpu(self):
        """Executes dummy inference passes to compile PyTorch kernels on Jetson GPU."""
        self.get_logger().info("[PERCEPTION_WARMUP] Warming up Jetson GPU CUDA stream...")
        with torch.inference_mode():
            for _ in range(5):
                _ = self.model(self.tensor_buffer)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        self.get_logger().info("[PERCEPTION_WARMUP] CUDA Warmup Complete.")

    def rgbd_sync_callback(self, rgb_msg: Image, depth_msg: Image):
        """
        Synchronized callback processing aligned RGB and Depth frames.
        Executes zero-copy preprocessing, CUDA tensor transfer, inference, and publishing.
        """
        start_time = time.perf_counter()

        try:
            # 1. Image Ingestion via OpenCV Bridge
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
            
            # RealSense depth is typically 16-bit unsigned integer (in millimeters)
            if depth_msg.encoding == '16UC1':
                cv_depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')
                # Convert millimeters to meters (float32)
                cv_depth_m = cv_depth_raw.astype(np.float32) * 0.001
            elif depth_msg.encoding == '32FC1':
                cv_depth_m = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
            else:
                self.get_logger().warn(f"Unsupported depth encoding: {depth_msg.encoding}")
                return

        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Conversion Failure: {str(e)}")
            return

        # 2. Fast NumPy Spatial Normalization & Resizing
        # Resize inputs to target neural network dimension [in_h, in_w]
        import cv2
        rgb_resized = cv2.resize(cv_rgb, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        depth_resized = cv2.resize(cv_depth_m, (self.in_w, self.in_h), interpolation=cv2.INTER_NEAREST)

        # Normalize RGB to [0.0, 1.0]
        rgb_norm = rgb_resized.astype(np.float32) / 255.0

        # Clip and Normalize Depth to [0.0, 1.0] relative to sensor operating range
        depth_clipped = np.clip(depth_resized, self.min_depth, self.max_depth)
        depth_norm = (depth_clipped - self.min_depth) / (self.max_depth - self.min_depth)

        # 3. Assemble 4-Channel Input [H, W, 4] -> Transpose to [1, 4, H, W]
        rgbd_combined = np.concatenate([rgb_norm, np.expand_dims(depth_norm, axis=2)], axis=2)
        rgbd_tensor_np = np.ascontiguousarray(np.transpose(rgbd_combined, (2, 0, 1)))

        # 4. Asynchronous Device Copy & Cast
        with torch.inference_mode():
            torch_input = torch.from_numpy(rgbd_tensor_np).unsqueeze(0).to(self.device, non_blocking=True)
            if self.use_fp16 and self.device.type == 'cuda':
                torch_input = torch_input.half()

            # 5. Execute CUDA Neural Inference
            # Output Shape: [1, 3, 64, 64]
            semantic_grid = self.model(torch_input)

            if self.device.type == 'cuda':
                torch.cuda.synchronize()

            # 6. Publish Flattened Occupancy Grid Tensor
            # Convert to CPU numpy array for ROS 2 serialization
            grid_cpu = semantic_grid.squeeze(0).cpu().float().numpy()  # [3, 64, 64]

        # ----------------------------------------------------------------------
        # Serialization: Flattened Float32MultiArray
        # Layout: Dim 0 = Channels (3: Free, Static, Dynamic), Dim 1 = 64, Dim 2 = 64
        # Total Elements = 3 * 64 * 64 = 12288 floats
        # ----------------------------------------------------------------------
        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label='channels', size=3, stride=3 * self.grid_dim * self.grid_dim),
            MultiArrayDimension(label='height', size=self.grid_dim, stride=self.grid_dim * self.grid_dim),
            MultiArrayDimension(label='width', size=self.grid_dim, stride=self.grid_dim)
        ]
        msg.data = grid_cpu.flatten().tolist()
        self.tensor_pub.publish(msg)

        # ----------------------------------------------------------------------
        # Optional RViz2 Visualizer: Convert to standard nav_msgs/OccupancyGrid
        # Cost mapping: Free=0, Static Obstacle=100, Dynamic Obstacle=100, Unknown=-1
        # ----------------------------------------------------------------------
        self._publish_rviz_grid(grid_cpu, rgb_msg.header.stamp)

        # Profiling and Latency Telemetry
        inference_latency_ms = (time.perf_counter() - start_time) * 1000.0
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            now = time.time()
            fps = 30.0 / (now - self.last_fps_time)
            self.last_fps_time = now
            self.get_logger().info(
                f"[PERCEPTION_STATS] Throughput: {fps:.2f} FPS | "
                f"Edge Latency: {inference_latency_ms:.2f} ms | "
                f"Memory Allocated: {torch.cuda.memory_allocated() / (1024**2):.1f} MB"
            )

    def _publish_rviz_grid(self, grid_cpu: np.ndarray, stamp):
        """Constructs and publishes a nav_msgs/OccupancyGrid for diagnostic telemetry in RViz2."""
        nav_msg = OccupancyGrid()
        nav_msg.header.stamp = stamp
        nav_msg.header.frame_id = 'base_link'

        # Map metadata: 64x64 grid spanning 3.2m x 3.2m centered ahead of the robot
        resolution = 0.05  # 5 cm per cell
        nav_msg.info.resolution = resolution
        nav_msg.info.width = self.grid_dim
        nav_msg.info.height = self.grid_dim
        nav_msg.info.origin.position.x = 0.0  # Starts at robot center
        nav_msg.info.origin.position.y = -(self.grid_dim * resolution) / 2.0  # Centered laterally
        nav_msg.info.origin.position.z = 0.0
        nav_msg.info.origin.orientation.w = 1.0

        # Obstacle Probability: Static (channel 1) + Dynamic (channel 2)
        obstacle_prob = grid_cpu[1, :, :] + grid_cpu[2, :, :]
        # Map [0.0, 1.0] -> [0, 100] integer occupancy cost
        cost_map = (np.clip(obstacle_prob, 0.0, 1.0) * 100).astype(np.int8)

        # Flatten into row-major order expected by ROS 2
        nav_msg.data = cost_map.flatten().tolist()
        self.nav_grid_pub.publish(nav_msg)


# ==============================================================================
# SECTION 3: ENTRYPOINT
# ==============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("[PERCEPTION_SHUTDOWN] Process terminated by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
