"""
하이브리드 제어기 - SEQ별 분리 제어

SEQ 1, 3: A* + PID (전역 경로 추종)
    - A*로 전역 경로 생성
    - PID로 경로 추종
    - SEQ별 다른 obstacle_margin 사용

SEQ 4: 순수 DWA (실시간 장애물 회피)
    - A* 경로 없음
    - 목적지 방향 lookahead 타겟 설정
    - /update_obstacle의 장애물 사각형 기반 가상 라이다
    - DWA로 실시간 장애물 회피
"""
import math
import time
import numpy as np
from controllers.pid_controller import PIDController
from planners.astar_planner import ObstacleRect
from planners.dwa_planner import DWAConfig, calc_dynamic_window, predict_trajectory, calc_to_goal_cost
from utils.visualization import save_path_image


class HybridController:
    """
    SEQ별 분리 제어기
    
    - SEQ 1, 3: A* + PID
    - SEQ 4: 순수 DWA (가상 라이다 기반)
    """
    
    def __init__(self, config, planner, state_manager):
        self.config = config
        self.planner = planner  # A* 플래너 (SEQ 1, 3용)
        self.state = state_manager
        
        # DWA 설정 (SEQ 4용)
        self.dwa_config = DWAConfig(config)
        
        # PID 제어기 (SEQ 1, 3용)
        self.steering_pid = PIDController(
            kp=config.PID.KP, 
            ki=config.PID.KI, 
            kd=config.PID.KD
        )
        
        # 상태 변수
        self.last_velocity = 0.0
        self.last_yaw_rate = 0.0
        self.stuck_counter = 0
        self.last_position = None
        
        # Stuck 복구 상태
        self.recovery_mode = False
        self.recovery_start_time = 0
        self.recovery_direction = 1
        
        # 디버그 카운터
        self._compute_count = 0

        # 경사도 적용
        self.slope_map = self._load_slope_map("slope_map.npy")
        self.default_max_speed = config.DWA.MAX_SPEED
        
    def reset(self):
        """제어기 상태 초기화"""
        self.steering_pid.reset()
        self.last_velocity = 0.0
        self.last_yaw_rate = 0.0
        self.stuck_counter = 0
        self.last_position = None
        self.recovery_mode = False
        
    def compute_action(self, curr_x, curr_z, curr_yaw):
        """메인 제어 루프"""
        
        # 1. 위치 업데이트
        curr_x, curr_z = self.config.clamp_world_xz(curr_x, curr_z)
        self.state.update_robot_pose(curr_x, curr_z)
        
        # SEQ 1, 3에서 obstacle_margin 업데이트
        if self.state.seq in [1, 3]:
            self._update_obstacle_margin()
        
        # 디버깅
        self._compute_count += 1
        if self._compute_count % 50 == 1:
            print(f"🚗 [compute_action] #{self._compute_count} SEQ={self.state.seq} "
                  f"pos=({curr_x:.1f},{curr_z:.1f}) dest={self.state.destination}")
        
        # 2. SEQ 2 사격 처리
        if self.state.seq == 2:
            cmd = self._stop_command()
            cmd["fire"] = True
            self.state.seq = 3
            self.state.status_message = "🔥 사격 완료! 다음 목적지를 선택하세요 (SEQ 3)"
            return cmd

        # 3. 목적지 없으면 정지
        if self.state.destination is None:
            return self._stop_command()
        
        # 4. 도착 확인 및 SEQ 전환
        dist_to_goal = math.hypot(
            self.state.destination[0] - curr_x, 
            self.state.destination[1] - curr_z
        )
        
        if dist_to_goal < self.config.ARRIVAL_THRESHOLD:
            return self._handle_arrival(curr_x, curr_z)

        # 5. Stuck 감지
        self._detect_stuck(curr_x, curr_z)
        
        # 6. Stuck 복구 모드 처리
        if self.stuck_counter >= self.config.Stuck.STUCK_COUNT_LIMIT:
            return self._recovery_action(curr_x, curr_z, curr_yaw)
        
        # 7. SEQ에 따른 제어 분기
        if self.state.seq == 4:
            # SEQ 4: 순수 DWA
            return self._seq4_pure_dwa(curr_x, curr_z, curr_yaw)
        else:
            # SEQ 1, 3: A* + PID
            return self._seq13_astar_pid(curr_x, curr_z, curr_yaw)
    
    def _load_slope_map(self, filename):
        """저장된 경사도 지도를 불러오고 결측치를 처리"""
        try:
            m = np.load(filename)
            # 데이터가 없는(NaN) 구역은 안전하게 평지(0도)로 가정하거나 평균값으로 대체
            m = np.nan_to_num(m, nan=0.0)
            print(f"✅ 경사도 지도 로드 완료 (Shape: {m.shape})")
            return m
        except:
            print("⚠️ 경사도 지도를 찾을 수 없어 기본 속도를 사용합니다.")
            return None

    def _update_obstacle_margin(self):
        """현재 SEQ에 맞는 obstacle_margin 적용"""
        if self.state.seq == 4:
            new_margin = self.config.ASTAR.OBSTACLE_MARGIN_SEQ4
        else:
            new_margin = self.config.ASTAR.get_obstacle_margin(self.state.seq)
        
        if new_margin != self.planner.obstacle_margin:
            self.planner.set_obstacle_margin(new_margin)
            print(f"🔧 SEQ {self.state.seq}: obstacle_margin = {new_margin}")
        
    def _handle_arrival(self, curr_x, curr_z):
        """도착 처리 및 SEQ 전환"""
        dist_to_goal = math.hypot(
            self.state.destination[0] - curr_x, 
            self.state.destination[1] - curr_z
        )
        print(f"✅ 도착! 거리={dist_to_goal:.2f}m (임계값={self.config.ARRIVAL_THRESHOLD}m)")
        
        if self.state.seq == 1:
            self.state.seq = 2
            self.state.status_message = "🎯 SEQ 1 도착! 사격 시스템 가동 중..."
            self.state.clear_path()
            self.state.destination = None
            print("🔄 SEQ 1→2 전환")
            return self._stop_command()
            
        elif self.state.seq == 3:
            self.state.seq = 4
            self.state.status_message = "🚀 SEQ 3 도착! 순수 DWA 모드 활성화"
            self.state.clear_path()
            self.state.destination = None
            print("🔄 SEQ 3→4 전환, 순수 DWA 시작")
            return self._stop_command()
            
        elif self.state.seq == 4:
            self.state.status_message = "🏁 모든 임무 완료!"
            self.state.clear_path()
            self.state.destination = None
            print("🏁 SEQ 4 완료!")
            return self._stop_command()
        
        else:
            self.state.clear_path()
            self.state.destination = None
            return self._stop_command()
    
    # ==================== SEQ 1, 3: A* + PID ====================
    
    def _seq13_astar_pid(self, curr_x, curr_z, curr_yaw):
        """SEQ 1, 3: A* 경로 + PID 제어"""
        
        # 경로가 없으면 생성
        if not self.state.global_path:
            self._generate_astar_path(curr_x, curr_z)
            if not self.state.global_path:
                print("⚠️ A* 경로 생성 실패")
                return self._stop_command()
        
        # 경로 업데이트 (지나간 노드 제거)
        self._update_path(curr_x, curr_z)
        
        # 타겟 포인트 선택
        target_point, _ = self._select_target_point(curr_x, curr_z)
        if not target_point:
            return self._stop_command()
        
        # PID 제어
        return self._pid_control(curr_x, curr_z, curr_yaw, target_point)
    
    def _generate_astar_path(self, curr_x, curr_z):
        """A* 경로 생성"""
        if self.state.destination is None:
            return
        
        dest_x, dest_z = self.state.destination

        mask_zones = []

        if self.state.seq == 1:
            forbidden_zone = ObstacleRect.from_min_max(158.0, 190.0, 115.0, 156.0)
            mask_zones.append(forbidden_zone)
            self.planner.update_grid_range(65.0, 200.0, 0.0, 300.0)
        elif self.state.seq == 3:
            self.planner.update_grid_range(0.0, 200.0, 150.0, 300.0)
        
        self.planner.set_mask_zones(mask_zones)
        
        path = self.planner.find_path(
            start=(curr_x, curr_z),
            goal=(dest_x, dest_z),
            use_obstacles=True
        )
        
        if path:
            self.state.global_path = path
            self.state.global_path_version += 1
            print(f"✅ A* 경로 생성: {len(path)}개 노드 (SEQ {self.state.seq})")
            # 경로 이미지 저장
            try:
                obs_count = len(self.planner._obstacles) if self.planner._obstacles else 0
                mode_label = f"A* + PID (SEQ {self.state.seq})"
                
                save_path_image(
                    planner=self.planner,
                    path=path,
                    current_pos=(curr_x, curr_z),
                    current_yaw=self.state.robot_yaw_deg,
                    filename=f"SEQ {self.state.seq}_Global_Path.png",
                    title=f"SEQ {self.state.seq} - {mode_label}",
                    state_manager=self.state
                )
                print(f"💾 경로 이미지 저장 완료: SEQ {self.state.seq}_path_debug.png ({len(path)}개 노드, 장애물 {obs_count}개)")
            except Exception as e:
                print(f"⚠️ 디버그 이미지 저장 실패: {e}")
        else:
            print(f"❌ A* 경로 생성 실패 (SEQ {self.state.seq})")
    
    def _update_path(self, curr_x, curr_z):
        """경로 업데이트: 지나간 노드 제거"""
        if not self.state.global_path:
            return
        
        # 현재 위치에서 가장 가까운 경로 노드 찾기
        min_dist = float('inf')
        closest_idx = 0
        
        for i, point in enumerate(self.state.global_path):
            dist = math.hypot(point[0] - curr_x, point[1] - curr_z)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # 지나간 노드 제거
        if closest_idx > 0:
            self.state.global_path = self.state.global_path[closest_idx:]
    
    def _select_target_point(self, curr_x, curr_z):
        """Lookahead 거리에 맞는 타겟 포인트 선택"""
        if not self.state.global_path:
            return None, 0
        
        lookahead = self.config.LOOKAHEAD_DIST
        cumulative_dist = 0.0
        prev_point = (curr_x, curr_z)
        
        for i, point in enumerate(self.state.global_path):
            segment_dist = math.hypot(
                point[0] - prev_point[0],
                point[1] - prev_point[1]
            )
            cumulative_dist += segment_dist
            
            if cumulative_dist >= lookahead:
                return point, i
            
            prev_point = point
        
        # 경로 끝에 도달하면 마지막 포인트 반환
        return self.state.global_path[-1], len(self.state.global_path) - 1
    
    def _pid_control(self, curr_x, curr_z, curr_yaw, target_node):
        """PID 조향 제어"""
        # 타겟 방향 계산
        dx = target_node[0] - curr_x
        dz = target_node[1] - curr_z
        target_angle_deg = math.degrees(math.atan2(dx, dz))
        
        # 각도 오차 계산
        error = target_angle_deg - curr_yaw
        while error > 180: 
            error -= 360
        while error < -180: 
            error += 360
        
        # PID 계산
        pid_output = self.steering_pid.compute(error)
        
        # 조향 가중치
        steer_weight = min(abs(pid_output), 1.0)
        steer_dir = "D" if pid_output > 0 else "A"
        if pid_output == 0: 
            steer_dir = ""
        
        # 속도 계산 (조향에 따른 감속)
        max_w = self.config.PID.MAX_SPEED_WEIGHT
        min_w = self.config.PID.MIN_SPEED_WEIGHT
        gain = self.config.PID.SPEED_REDUCT_GAIN
        error_th = self.config.PID.ERROR_THRESHOLD
        error_range = self.config.PID.ERROR_RANGE

        speed_weight = max(min_w, max_w - steer_weight * gain)
        
        if abs(error) > error_th:
            reduction_factor = max(0.0, 1.0 - (abs(error) - error_th) / error_range)
            speed_weight *= reduction_factor
        speed_weight = max(speed_weight, min_w)
        
        if speed_weight <= 0.05:
            cmd_ws = "STOP"
            speed_weight = 1.0
        else:
            cmd_ws = "W"
        
        return {
            "moveWS": {"command": cmd_ws, "weight": round(speed_weight, 2)},
            "moveAD": {"command": steer_dir, "weight": round(steer_weight * self.config.PID.STEER_SENSITIVITY, 2)},
            "fire": False
        }
    
    # ==================== SEQ 4: 순수 DWA ====================
    
    def _seq4_pure_dwa(self, curr_x, curr_z, curr_yaw):
        """SEQ 4: 순수 DWA 제어 (A* 없음, 가상 라이다 기반) + 경사도에 따른 속도 가변 적용"""
        
        # 1. 현재 위치의 경사도 확인
        gx, gz = int(curr_x), int(curr_z)
        current_slope = 0.0
        if self.slope_map is not None and 0 <= gx < 300 and 0 <= gz < 300:
            current_slope = self.slope_map[gz, gx]

        # 2. 경사도에 따른 Dynamic Max Speed 설정
        if current_slope > 30.0:
            self.dwa_config.max_speed = self.default_max_speed * 0.3 # 급경사
        elif current_slope > 15.0:
            self.dwa_config.max_speed = self.default_max_speed * 0.5 # 완경사
        else:
            self.dwa_config.max_speed = self.default_max_speed       # 평지
            
        if self.state.destination is None:
            return self._stop_command()
        
        # 목적지 방향으로 lookahead 타겟 계산
        target_point = self._calc_lookahead_target(curr_x, curr_z)
        
        if self._compute_count % 20 == 1:
            print(f"🎯 [SEQ4 DWA] pos=({curr_x:.1f},{curr_z:.1f}) → "
                  f"target=({target_point[0]:.1f},{target_point[1]:.1f}) → "
                  f"dest=({self.state.destination[0]:.1f},{self.state.destination[1]:.1f}), "
                  f"obstacles={len(self.state.obstacle_rects)}개")
        
        # DWA 제어 (가상 라이다 비용 사용)
        return self._dwa_control_virtual_lidar(curr_x, curr_z, curr_yaw, target_point)
    
    def _calc_lookahead_target(self, curr_x, curr_z):
        """목적지 방향으로 lookahead 거리만큼의 타겟 계산"""
        dest_x, dest_z = self.state.destination
        
        # 목적지까지의 거리와 방향
        dx = dest_x - curr_x
        dz = dest_z - curr_z
        dist_to_dest = math.hypot(dx, dz)
        
        # SEQ 4 전용 lookahead 사용
        lookahead = self.config.SEQ4.LOOKAHEAD_DIST
        
        # 목적지가 lookahead보다 가까우면 목적지 그대로 사용
        if dist_to_dest <= lookahead:
            return (dest_x, dest_z)
        
        # 목적지 방향으로 lookahead 거리만큼의 타겟
        ratio = lookahead / dist_to_dest
        target_x = curr_x + dx * ratio
        target_z = curr_z + dz * ratio
        
        return (target_x, target_z)
    
    def _dwa_control_virtual_lidar(self, curr_x, curr_z, curr_yaw, target_point):
        """DWA 제어 - 가상 라이다 기반 장애물 비용"""
        
        curr_yaw_rad = math.radians(curr_yaw)
        x = np.array([curr_x, curr_z, curr_yaw_rad, self.last_velocity, self.last_yaw_rate])
        
        # Dynamic Window 계산
        dw = calc_dynamic_window(x, self.dwa_config)
        
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x])
        valid_trajectories = 0
        total_trajectories = 0
        
        obstacle_margin = self.config.ASTAR.OBSTACLE_MARGIN_SEQ4
        
        # 모든 (v, omega) 조합 탐색
        for v in np.arange(dw[0], dw[1], self.dwa_config.v_resolution):
            for omega in np.arange(dw[2], dw[3], self.dwa_config.yaw_rate_resolution):
                total_trajectories += 1
                trajectory = predict_trajectory(x, v, omega, self.dwa_config)
                
                # 1. 월드 경계 체크
                out_of_bounds = False
                for state in trajectory:
                    tx, tz = state[0], state[1]
                    if (tx < self.config.WORLD_MIN_XZ or tx > self.config.WORLD_MAX_XZ or
                        tz < self.config.WORLD_MIN_XZ or tz > self.config.WORLD_MAX_XZ):
                        out_of_bounds = True
                        break
                
                if out_of_bounds:
                    continue
                
                # 2. 가상 라이다 기반 장애물 비용 계산
                ob_cost = self._calc_virtual_lidar_cost(trajectory, obstacle_margin)
                if ob_cost == float("inf"):
                    continue  # 충돌 경로는 제외
                
                # 3. 목표 비용
                to_goal_cost = self.dwa_config.to_goal_cost_gain * calc_to_goal_cost(
                    trajectory, [target_point[0], target_point[1]]
                )
                
                # 4. 속도 비용 (빠를수록 좋음)
                speed_cost = self.dwa_config.speed_cost_gain * (
                    self.dwa_config.max_speed - trajectory[-1, 3]
                )
                
                # 5. 조향 패널티
                steering_penalty = abs(omega) * self.dwa_config.steering_penalty
                
                # 총 비용
                final_cost = to_goal_cost + speed_cost + ob_cost + steering_penalty
                
                valid_trajectories += 1
                
                if final_cost < min_cost:
                    min_cost = final_cost
                    best_u = [v, omega]
                    best_trajectory = trajectory
        
        # DWA 결과 로깅
        self.state.valid_traj_count = valid_trajectories
        if self._compute_count % 10 == 1:
            print(f"🎯 DWA: 총={total_trajectories}, 유효={valid_trajectories}, "
                  f"비용={min_cost:.2f}, v={best_u[0]:.2f}, ω={best_u[1]:.3f}")
        
        # 유효 경로 없음 → 후진 시도
        if valid_trajectories == 0:
            print("⚠️ DWA 유효 경로 없음 → 후진 시도")
            return {
                "moveWS": {"command": "S", "weight": 0.3},
                "moveAD": {"command": "", "weight": 0.0},
                "fire": False
            }
        
        # DWA 궤적 저장 (시각화용)
        self.state.last_dwa_traj = best_trajectory
        self.state.last_dwa_target = (float(target_point[0]), float(target_point[1]))
        self.state.local_traj_version += 1
        
        # 속도 업데이트
        desired_v = float(best_u[0])
        desired_omega = float(best_u[1])
        
        # Stuck 방지
        if (abs(desired_v) < self.dwa_config.robot_stuck_flag_cons and 
            abs(x[3]) < self.dwa_config.robot_stuck_flag_cons):
            desired_v = -float(self.config.Recovery.REVERSE_SPEED)
            desired_omega = 0.0
        
        self.last_velocity = desired_v
        self.last_yaw_rate = desired_omega
        
        # 명령어 변환
        steer_command = desired_omega / self.dwa_config.max_yaw_rate
        steer_command = max(min(steer_command, 1.0), -1.0)
        steer_weight = abs(steer_command)
        
        if abs(steer_command) < 0.05:
            steer_dir = ""
            steer_weight = 0.0
        else:
            steer_dir = "D" if steer_command > 0 else "A"
        
        ws_cmd = "W" if desired_v > 0.05 else ("S" if desired_v < -0.05 else "STOP")
        ws_weight = min(max(abs(desired_v) / self.dwa_config.max_speed, 0.0), 1.0)
        
        return {
            "moveWS": {"command": ws_cmd, "weight": round(ws_weight, 2)},
            "moveAD": {"command": steer_dir, "weight": round(steer_weight, 2)},
            "fire": False
        }
    
    def _calc_virtual_lidar_cost(self, trajectory, obstacle_margin):
        """가상 라이다 기반 장애물 비용 계산
        
        - 장애물 사각형(obstacle_rects)과의 거리를 기반으로 비용 계산
        - 충돌(collision_distance 이내) → inf
        - 위험(danger_distance 이내) → 높은 비용
        - 안전(safe_distance 이상) → 낮은 비용
        """
        collision_dist = self.config.DWA.COLLISION_DISTANCE
        danger_dist = self.config.DWA.DANGER_DISTANCE
        safe_dist = self.config.DWA.SAFE_DISTANCE
        
        total_cost = 0.0
        min_dist_overall = float('inf')
        
        # 궤적의 각 포인트에서 장애물 거리 체크
        for i, state in enumerate(trajectory):
            if i < 3:  # 처음 몇 포인트는 스킵 (현재 위치 근처)
                continue
            
            px, pz = state[0], state[1]
            
            # 가장 가까운 장애물까지의 거리
            dist = self.state.get_obstacle_distance(px, pz, obstacle_margin)
            
            if dist < min_dist_overall:
                min_dist_overall = dist
            
            # 충돌 거리 이내 → 무효 경로
            if dist <= collision_dist:
                return float("inf")
        
        # 거리 기반 비용 계산
        if min_dist_overall <= danger_dist:
            # 위험 구간: 높은 비용
            normalized = (min_dist_overall - collision_dist) / max(danger_dist - collision_dist, 0.1)
            total_cost = 50.0 * (1.0 - normalized)
        elif min_dist_overall <= safe_dist:
            # 주의 구간: 중간 비용
            normalized = (min_dist_overall - danger_dist) / max(safe_dist - danger_dist, 0.1)
            total_cost = 10.0 * (1.0 - normalized)
        else:
            # 안전 구간: 낮은 비용
            total_cost = 0.0
        
        return total_cost * self.dwa_config.obstacle_cost_gain
    
    # ==================== Stuck 감지/복구 ====================
    
    def _detect_stuck(self, curr_x, curr_z):
        """Stuck 감지"""
        if self.last_position is None:
            self.last_position = (curr_x, curr_z)
            return
        
        dist = math.hypot(
            curr_x - self.last_position[0],
            curr_z - self.last_position[1]
        )
        
        if dist < self.config.Stuck.STUCK_THRESHOLD:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        
        self.last_position = (curr_x, curr_z)
    
    def _recovery_action(self, curr_x, curr_z, curr_yaw):
        """Stuck 복구 동작"""
        rc = self.config.Recovery
        
        if not self.recovery_mode:
            self.recovery_mode = True
            self.recovery_start_time = time.time()
            self.recovery_direction = 1 if (self.stuck_counter % 2 == 0) else -1
            print(f"🔧 복구 시작: {'좌회전' if self.recovery_direction > 0 else '우회전'} 후진")
        
        elapsed = time.time() - self.recovery_start_time
        
        if elapsed < rc.PHASE1_SEC:
            # Phase 1: 후진 + 회전
            return {
                "moveWS": {"command": "S", "weight": rc.PHASE1_WS_WEIGHT},
                "moveAD": {"command": "D" if self.recovery_direction > 0 else "A", 
                          "weight": rc.PHASE1_AD_WEIGHT},
                "fire": False
            }
        
        elif elapsed < rc.PHASE1_SEC + rc.PHASE2_SEC:
            # Phase 2: 제자리 회전
            return {
                "moveWS": {"command": "STOP", "weight": 1.0},
                "moveAD": {"command": "D" if self.recovery_direction > 0 else "A", 
                          "weight": rc.PHASE2_AD_WEIGHT},
                "fire": False
            }
        
        else:
            # 복구 완료
            print("✅ 복구 완료!")
            self.recovery_mode = False
            self.stuck_counter = 0
            self.last_position = None
            self.state.clear_path()  # 경로 재생성 유도
            return self._stop_command()
    
    @staticmethod
    def _stop_command():
        """정지 명령"""
        return {
            "moveWS": {"command": "STOP", "weight": 1.0},
            "moveAD": {"command": "", "weight": 0.0}, 
            "fire": False
        }