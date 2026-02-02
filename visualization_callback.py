"""
visualization_callback.py

v9 스타일 시각화 콜백
- 주기적으로 별도 환경에서 Pygame 창 띄워서 에피소드 관전
- 학습 환경(SubprocVecEnv)과 분리되어 충돌 없음
"""

import os
import numpy as np
import pygame
import math
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional


class VisualEvalCallback(BaseCallback):
    """
    v9 스타일 시각화 평가 콜백
    
    - eval_freq 스텝마다 Pygame 창 열어서 n_eval_episodes 에피소드 관전
    - 학습 환경과 별도의 eval_env 사용 (SubprocVecEnv 충돌 방지)
    """
    
    def __init__(
        self,
        eval_env,  # TankNavEnv 인스턴스 (render_mode=None으로 생성)
        eval_freq: int = 25000,
        n_eval_episodes: int = 3,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        
        # Pygame 설정
        self.screen = None
        self.clock = None
        self.font = None
        self.window_size = 850
        self.scale = self.window_size / 300.0  # map_size=300 가정
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0 and self.n_calls > 0:
            self._run_visual_eval()
        return True
    
    def _run_visual_eval(self):
        """Pygame 창 띄워서 에피소드 관전"""
        if self.verbose:
            print(f"\n🎮 Visual evaluation at step {self.n_calls}...")
        
        # Pygame 초기화
        self._init_pygame()
        
        total_rewards = []
        successes = 0
        
        for ep in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            truncated = False
            episode_reward = 0
            step = 0
            
            while not (done or truncated):
                # 모델 추론
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                episode_reward += reward
                step += 1
                
                # Pygame 렌더링
                self._render_pygame(info, episode_reward, ep + 1, step)
                
                # 이벤트 처리 (창 닫기 등)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self._close_pygame()
                        return
                
                # FPS 제한
                self.clock.tick(30)
            
            total_rewards.append(episode_reward)
            if info.get('reached_goal', False):
                successes += 1
        
        # 결과 출력
        if self.verbose:
            mean_reward = np.mean(total_rewards)
            success_rate = successes / self.n_eval_episodes * 100
            print(f"   ✅ Mean Reward: {mean_reward:.1f}, Success: {successes}/{self.n_eval_episodes} ({success_rate:.0f}%)")
        
        # Pygame 종료
        self._close_pygame()
    
    def _init_pygame(self):
        """Pygame 초기화"""
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption(f"Tank RL - Step {self.n_calls}")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 16)
    
    def _close_pygame(self):
        """Pygame 종료"""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
    
    def _coord(self, x, z):
        """월드 좌표 → 스크린 좌표"""
        px = int(x * self.scale)
        py = int(self.window_size - (z * self.scale))
        return px, py
    
    def _render_pygame(self, info, episode_reward, episode_num, step):
        """Pygame 렌더링"""
        env = self.eval_env
        
        # 배경
        self.screen.fill((240, 240, 240))
        
        # 장애물
        if hasattr(env, 'obstacle_rects'):
            for obs in env.obstacle_rects:
                if isinstance(obs, tuple) and len(obs) == 4:
                    x_min, x_max, z_min, z_max = obs
                else:
                    continue
                
                p1 = self._coord(x_min, z_max)
                p2 = self._coord(x_max, z_min)
                w = max(1, p2[0] - p1[0])
                h = max(1, p2[1] - p1[1])
                
                # 크기별 색상
                area = (x_max - x_min) * (z_max - z_min)
                if area < 10:
                    color = (34, 139, 34)  # 녹색
                elif area < 100:
                    color = (139, 90, 43)  # 갈색
                else:
                    color = (80, 80, 80)   # 회색
                
                pygame.draw.rect(self.screen, color, pygame.Rect(p1[0], p1[1], w, h))
        
        # 경로
        if hasattr(env, 'path') and env.path and len(env.path) > 1:
            points = [self._coord(p[0], p[1]) for p in env.path]
            pygame.draw.lines(self.screen, (0, 0, 200), False, points, 2)
        
        # 궤적
        if hasattr(env, 'trajectory') and len(env.trajectory) > 1:
            points = [self._coord(p[0], p[1]) for p in env.trajectory[-200:]]
            if len(points) > 1:
                pygame.draw.lines(self.screen, (0, 180, 0), False, points, 2)
        
        # 목표
        if hasattr(env, 'goal') and env.goal:
            gx, gy = self._coord(env.goal[0], env.goal[1])
            pygame.draw.circle(self.screen, (255, 50, 50), (gx, gy), 12)
            pygame.draw.circle(self.screen, (200, 0, 0), (gx, gy), 6)
        
        # 타겟
        if hasattr(env, 'target') and env.target:
            tx, ty = self._coord(env.target[0], env.target[1])
            pygame.draw.circle(self.screen, (0, 100, 255), (tx, ty), 6)
        
        # 전차
        if hasattr(env, 'state') and env.state:
            state = env.state
            tank_x, tank_y = self._coord(state.x, state.z)
            
            # 라이다 그리기
            if hasattr(env, '_cast_lidar_vectorized'):
                lidar = env._cast_lidar_vectorized()
                num_rays = len(lidar)
                yaw_rad = math.radians(state.yaw)
                
                for i, dist in enumerate(lidar):
                    angle = (i / num_rays) * 2 * math.pi + yaw_rad
                    
                    # 안전 거리에 따른 색상
                    if hasattr(env, 'tank_boundary_dist'):
                        margin = dist - env.tank_boundary_dist[i]
                        if margin < env.config.margin_critical:
                            color = (255, 0, 0)
                        elif margin < env.config.margin_warning:
                            color = (255, 180, 0)
                        else:
                            color = (0, 200, 0)
                    else:
                        color = (0, 200, 0)
                    
                    end_x = state.x + dist * math.cos(angle)
                    end_z = state.z + dist * math.sin(angle)
                    pygame.draw.line(self.screen, color, (tank_x, tank_y), 
                                     self._coord(end_x, end_z), 1)
            
            # 전차 본체
            yaw_rad = math.radians(state.yaw)
            half_l = 4.0  # 시각적 크기
            half_w = 2.0
            
            corners = [
                (half_l, half_w), (half_l, -half_w),
                (-half_l, -half_w), (-half_l, half_w)
            ]
            rot_corners = []
            for lx, lz in corners:
                # yaw=0 → +z 방향이므로 sin/cos 조합 주의
                rx = lx * math.sin(yaw_rad) + lz * math.cos(yaw_rad)
                rz = lx * math.cos(yaw_rad) - lz * math.sin(yaw_rad)
                rot_corners.append(self._coord(state.x + rx, state.z + rz))
            
            pygame.draw.polygon(self.screen, (50, 120, 50), rot_corners)
            
            # 포탑
            pygame.draw.circle(self.screen, (30, 80, 30), (tank_x, tank_y), 6)
            
            # 포신
            cannon_len = 6.0
            cannon_x = state.x + math.sin(yaw_rad) * cannon_len
            cannon_z = state.z + math.cos(yaw_rad) * cannon_len
            pygame.draw.line(self.screen, (20, 40, 20), (tank_x, tank_y),
                             self._coord(cannon_x, cannon_z), 3)
        
        # 정보 패널
        if self.font and hasattr(env, 'state') and env.state:
            state = env.state
            dist_to_goal = info.get('distance_to_goal', 0)
            heading_err = info.get('heading_error', 0)
            
            lines = [
                f"Training Step {self.n_calls:,}",
                f"Episode {episode_num}/{self.n_eval_episodes}",
                f"",
                f"Pos: ({state.x:.1f}, {state.z:.1f})",
                f"Yaw: {state.yaw:.1f} deg",
                f"Speed: {state.speed:.1f} m/s",
                f"",
                f"Goal Dist: {dist_to_goal:.1f}m",
                f"Head Err: {heading_err:.1f} deg",
                f"",
                f"Ep Step: {step}",
                f"Ep Reward: {episode_reward:.1f}",
            ]
            
            # 상태 표시
            if info.get('collision', False):
                lines.append("")
                lines.append("!! COLLISION !!")
            elif info.get('reached_goal', False):
                lines.append("")
                lines.append("** GOAL REACHED **")
            
            for i, line in enumerate(lines):
                if "COLLISION" in line:
                    color = (255, 0, 0)
                elif "GOAL" in line:
                    color = (0, 150, 0)
                else:
                    color = (0, 0, 0)
                
                text = self.font.render(line, True, color)
                self.screen.blit(text, (10, 10 + i * 20))
        
        pygame.display.flip()


class ImageSaveCallback(BaseCallback):
    """
    주기적으로 스냅샷 이미지 저장 (headless)
    - Pygame 창 없이 이미지만 저장
    - SubprocVecEnv에서도 안전
    """
    
    def __init__(
        self,
        eval_env,
        save_path: str = "./viz",
        save_freq: int = 10000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.save_path = save_path
        self.save_freq = save_freq
        
        os.makedirs(save_path, exist_ok=True)
        
        # Surface for rendering
        self.window_size = 800
        self.scale = self.window_size / 300.0
        pygame.init()
        self.surface = pygame.Surface((self.window_size, self.window_size))
        self.font = pygame.font.SysFont("Arial", 14)
    
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0 and self.n_calls > 0:
            self._save_snapshot()
        return True
    
    def _save_snapshot(self):
        """스냅샷 저장"""
        # 1 에피소드 실행
        obs, _ = self.eval_env.reset()
        done = False
        truncated = False
        
        while not (done or truncated):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = self.eval_env.step(action)
        
        # 마지막 프레임 저장
        self._render_to_surface()
        
        result = "GOAL" if info.get('reached_goal') else ("CRASH" if info.get('collision') else "TIME")
        filename = os.path.join(self.save_path, f"step_{self.n_calls:08d}_{result}.png")
        pygame.image.save(self.surface, filename)
        
        if self.verbose:
            print(f"💾 Snapshot saved: {filename}")
    
    def _coord(self, x, z):
        px = int(x * self.scale)
        py = int(self.window_size - (z * self.scale))
        return px, py
    
    def _render_to_surface(self):
        """Surface에 렌더링"""
        env = self.eval_env
        self.surface.fill((240, 240, 240))
        
        # 장애물
        if hasattr(env, 'obstacle_rects'):
            for obs in env.obstacle_rects:
                if isinstance(obs, tuple) and len(obs) == 4:
                    x_min, x_max, z_min, z_max = obs
                    p1 = self._coord(x_min, z_max)
                    w = (x_max - x_min) * self.scale
                    h = (z_max - z_min) * self.scale
                    pygame.draw.rect(self.surface, (100, 100, 100), 
                                     pygame.Rect(p1[0], p1[1], w, h))
        
        # 목표
        if hasattr(env, 'goal') and env.goal:
            gx, gy = self._coord(env.goal[0], env.goal[1])
            pygame.draw.circle(self.surface, (255, 0, 0), (gx, gy), 8)
        
        # 궤적
        if hasattr(env, 'trajectory') and len(env.trajectory) > 1:
            points = [self._coord(p[0], p[1]) for p in env.trajectory]
            pygame.draw.lines(self.surface, (0, 180, 0), False, points, 2)
        
        # 전차
        if hasattr(env, 'state') and env.state:
            tx, ty = self._coord(env.state.x, env.state.z)
            pygame.draw.circle(self.surface, (50, 100, 50), (tx, ty), 8)
    
    def _on_training_end(self):
        pygame.quit()