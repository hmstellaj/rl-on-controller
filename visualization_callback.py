"""
visualization_callback.py

학습 중 시각화 콜백
- 전차 위치와 방향 (크기 반영)
- A* 경로와 waypoint 노드
- 가상 라이다 레이캐스팅
- 장애물 및 지형

[사용법]
    from visualization_callback import VisualizationCallback
    
    viz_callback = VisualizationCallback(
        env=env,
        save_path="./viz",
        save_freq=1000,
    )
    model.learn(callbacks=[viz_callback])
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Arrow, Wedge, Circle
from matplotlib.collections import LineCollection
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime


class VisualizationCallback(BaseCallback):
    """
    학습 중 시각화 콜백
    
    주기적으로 현재 에피소드 상태를 이미지로 저장
    """
    
    def __init__(
        self,
        save_path: str = "./viz",
        save_freq: int = 5000,
        episode_save_freq: int = 50,
        figsize: Tuple[int, int] = (14, 14),
        dpi: int = 100,
        show_lidar: bool = True,
        show_path: bool = True,
        show_trajectory: bool = True,
        show_obstacles: bool = True,
        show_heatmap: bool = False,
        verbose: int = 1,
    ):
        """
        Args:
            save_path: 이미지 저장 경로
            save_freq: 스텝 단위 저장 주기
            episode_save_freq: 에피소드 단위 저장 주기
            figsize: 그림 크기
            dpi: 해상도
            show_lidar: 라이다 레이 표시
            show_path: 경로 표시
            show_trajectory: 궤적 표시
            show_obstacles: 장애물 표시
            show_heatmap: 경사도 히트맵 표시
        """
        super().__init__(verbose)
        
        self.save_path = save_path
        self.save_freq = save_freq
        self.episode_save_freq = episode_save_freq
        self.figsize = figsize
        self.dpi = dpi
        
        self.show_lidar = show_lidar
        self.show_path = show_path
        self.show_trajectory = show_trajectory
        self.show_obstacles = show_obstacles
        self.show_heatmap = show_heatmap
        
        # 상태 추적
        self.episode_count = 0
        self.step_in_episode = 0
        self.current_trajectory = []
        self.episode_rewards = []
        self.current_episode_reward = 0
        
        # 저장 경로 생성
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, "episodes"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "steps"), exist_ok=True)
        
        # 전차 크기 (Unity 기준)
        self.tank_width = 3.667
        self.tank_length = 8.066
        
    def _on_training_start(self):
        """학습 시작 시"""
        self.episode_count = 0
        self.current_trajectory = []
        print(f"📊 시각화 콜백 활성화 (저장 경로: {self.save_path})")
        
    def _on_step(self) -> bool:
        """매 스텝마다 호출"""
        self.step_in_episode += 1
        
        # 현재 환경에서 상태 가져오기
        env = self.training_env.envs[0]
        if hasattr(env, 'env'):
            env = env.env  # Monitor wrapper 벗기기
        
        if hasattr(env, 'state') and env.state is not None:
            pos = (env.state.x, env.state.z)
            self.current_trajectory.append(pos)
            
        # 보상 추적
        if self.locals.get('rewards') is not None:
            self.current_episode_reward += self.locals['rewards'][0]
        
        # 주기적 스텝 저장
        if self.num_timesteps % self.save_freq == 0:
            self._save_step_visualization(env)
        
        # 에피소드 종료 체크
        if self.locals.get('dones') is not None and self.locals['dones'][0]:
            self._on_episode_end(env)
        
        return True
    
    def _on_episode_end(self, env):
        """에피소드 종료 시"""
        self.episode_count += 1
        self.episode_rewards.append(self.current_episode_reward)
        
        # 주기적 에피소드 저장
        if self.episode_count % self.episode_save_freq == 0:
            self._save_episode_visualization(env)
        
        # 상태 리셋
        self.current_trajectory = []
        self.current_episode_reward = 0
        self.step_in_episode = 0
    
    def _save_step_visualization(self, env):
        """스텝 단위 시각화 저장"""
        try:
            fig = self._create_visualization(env, title=f"Step {self.num_timesteps:,}")
            
            filename = os.path.join(
                self.save_path, "steps", 
                f"step_{self.num_timesteps:08d}.png"
            )
            fig.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            
            if self.verbose:
                print(f"💾 시각화 저장: {filename}")
                
        except Exception as e:
            print(f"⚠️ 시각화 저장 실패: {e}")
    
    def _save_episode_visualization(self, env):
        """에피소드 단위 시각화 저장"""
        try:
            # 결과 정보
            info = self.locals.get('infos', [{}])[0]
            reached_goal = info.get('reached_goal', False)
            collision = info.get('collision', False)
            
            status = "SUCCESS" if reached_goal else ("COLLISION" if collision else "TIMEOUT")
            title = f"Episode {self.episode_count} - {status} (Reward: {self.current_episode_reward:.1f})"
            
            fig = self._create_visualization(env, title=title, show_full_trajectory=True)
            
            filename = os.path.join(
                self.save_path, "episodes",
                f"episode_{self.episode_count:05d}_{status}.png"
            )
            fig.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            
            if self.verbose:
                print(f"💾 에피소드 시각화 저장: {filename}")
                
        except Exception as e:
            print(f"⚠️ 에피소드 시각화 저장 실패: {e}")
    
    def _create_visualization(
        self, 
        env, 
        title: str = "",
        show_full_trajectory: bool = False,
    ) -> plt.Figure:
        """시각화 생성"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 배경 설정
        ax.set_facecolor('#f0f0f0')
        
        # 1. 경사도 히트맵 (선택적)
        if self.show_heatmap and hasattr(env, 'slope_map') and env.slope_map is not None:
            self._draw_heatmap(ax, env)
        
        # 2. 장애물
        if self.show_obstacles and hasattr(env, 'obstacle_rects'):
            self._draw_obstacles(ax, env.obstacle_rects)
        
        # 3. 경로 & Waypoints
        if self.show_path and hasattr(env, 'path') and env.path:
            self._draw_path(ax, env.path, env)
        
        # 4. 궤적
        if self.show_trajectory and self.current_trajectory:
            self._draw_trajectory(ax, self.current_trajectory)
        
        # 5. 전차 (크기 반영)
        if hasattr(env, 'state') and env.state is not None:
            self._draw_tank(ax, env.state.x, env.state.z, env.state.yaw)
            
            # 6. 라이다 레이
            if self.show_lidar:
                self._draw_lidar(ax, env)
        
        # 7. 타겟 & 목표
        if hasattr(env, 'target') and env.target is not None:
            ax.plot(env.target[0], env.target[1], 'b^', markersize=12, 
                    label='Target', zorder=15)
        
        if hasattr(env, 'goal') and env.goal is not None:
            ax.plot(env.goal[0], env.goal[1], 'r*', markersize=20, 
                    label='Goal', zorder=15)
        
        # 8. 정보 패널
        self._draw_info_panel(ax, env)
        
        # 축 설정
        map_size = getattr(env, 'config', None)
        if map_size and hasattr(map_size, 'map_size'):
            ax.set_xlim(0, map_size.map_size)
            ax.set_ylim(0, map_size.map_size)
        else:
            ax.set_xlim(0, 300)
            ax.set_ylim(0, 300)
        
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Z (m)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        return fig
    
    def _draw_heatmap(self, ax, env):
        """경사도 히트맵"""
        slope_map = env.slope_map
        extent = [0, slope_map.shape[1], 0, slope_map.shape[0]]
        
        im = ax.imshow(
            slope_map, 
            extent=extent, 
            origin='lower',
            cmap='YlOrRd', 
            alpha=0.3, 
            aspect='auto',
            vmin=0, 
            vmax=45
        )
        # 컬러바는 너무 복잡해지므로 생략
    
    def _draw_obstacles(self, ax, obstacle_rects: List[Tuple]):
        """장애물 그리기"""
        for obs in obstacle_rects:
            if isinstance(obs, tuple) and len(obs) == 4:
                x_min, x_max, z_min, z_max = obs
            elif isinstance(obs, dict):
                x_min = obs['x_min']
                x_max = obs['x_max']
                z_min = obs['z_min']
                z_max = obs['z_max']
            else:
                continue
            
            width = x_max - x_min
            height = z_max - z_min
            
            # 장애물 크기에 따른 색상
            area = width * height
            if area < 5:
                color = '#228B22'  # 작은 것 (나무) - 녹색
                alpha = 0.6
            elif area < 50:
                color = '#8B4513'  # 중간 (바위) - 갈색
                alpha = 0.7
            else:
                color = '#4a4a4a'  # 큰 것 (건물) - 회색
                alpha = 0.8
            
            rect = patches.Rectangle(
                (x_min, z_min), width, height,
                linewidth=0.5,
                edgecolor='black',
                facecolor=color,
                alpha=alpha,
                zorder=5
            )
            ax.add_patch(rect)
    
    def _draw_path(self, ax, path: List[Tuple], env):
        """경로 & Waypoints"""
        if not path:
            return
        
        # 경로 선
        path_x = [p[0] for p in path]
        path_z = [p[1] for p in path]
        ax.plot(path_x, path_z, 'b-', linewidth=2, alpha=0.6, label='A* Path', zorder=8)
        
        # Waypoint 노드
        for i, (px, pz) in enumerate(path):
            # 현재 타겟 인덱스 이전/이후 구분
            current_idx = getattr(env, 'current_path_idx', 0)
            
            if i < current_idx:
                # 지나간 노드
                ax.plot(px, pz, 'o', color='gray', markersize=4, alpha=0.5, zorder=9)
            elif i == current_idx:
                # 현재 노드
                ax.plot(px, pz, 'o', color='blue', markersize=8, zorder=10)
            else:
                # 앞으로 갈 노드
                ax.plot(px, pz, 'o', color='cyan', markersize=5, alpha=0.7, zorder=9)
        
        # 시작점, 끝점 강조
        ax.plot(path_x[0], path_z[0], 'gs', markersize=12, label='Start', zorder=11)
    
    def _draw_trajectory(self, ax, trajectory: List[Tuple]):
        """궤적 그리기 (그라데이션)"""
        if len(trajectory) < 2:
            return
        
        # 색상 그라데이션 (오래된 것 → 최신)
        points = np.array(trajectory)
        segments = np.array([[points[i], points[i+1]] for i in range(len(points)-1)])
        
        # 색상 맵
        colors = plt.cm.viridis(np.linspace(0.2, 1, len(segments)))
        
        lc = LineCollection(segments, colors=colors, linewidth=2, alpha=0.8, zorder=7)
        ax.add_collection(lc)
    
    def _draw_tank(self, ax, x: float, z: float, yaw: float):
        """전차 그리기 (크기 반영, 방향 표시)"""
        
        # 전차 크기
        width = self.tank_width
        length = self.tank_length
        
        # 회전 변환
        yaw_rad = math.radians(yaw)
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)
        
        # 전차 몸체 (사각형) - 중심 기준 꼭짓점
        half_w = width / 2
        half_l = length / 2
        
        corners = [
            (-half_w, -half_l),  # 좌하
            (half_w, -half_l),   # 우하
            (half_w, half_l),    # 우상
            (-half_w, half_l),   # 좌상
        ]
        
        # 회전 적용 (Unity 좌표계: yaw=0 → +z)
        rotated_corners = []
        for cx, cz in corners:
            # x, z 좌표를 yaw만큼 회전
            rx = cx * cos_y + cz * sin_y
            rz = -cx * sin_y + cz * cos_y
            rotated_corners.append((x + rx, z + rz))
        
        # 전차 몸체
        tank_body = patches.Polygon(
            rotated_corners,
            closed=True,
            facecolor='#2E8B57',  # 군용 녹색
            edgecolor='black',
            linewidth=2,
            zorder=12,
            label='Tank'
        )
        ax.add_patch(tank_body)
        
        # 포탑 (원형)
        turret = Circle(
            (x, z), 
            radius=width * 0.35,
            facecolor='#3CB371',
            edgecolor='black',
            linewidth=1.5,
            zorder=13
        )
        ax.add_patch(turret)
        
        # 포신 (전방 방향)
        barrel_length = length * 0.5
        barrel_end_x = x + sin_y * barrel_length
        barrel_end_z = z + cos_y * barrel_length
        
        ax.plot(
            [x, barrel_end_x], 
            [z, barrel_end_z],
            color='#1a1a1a',
            linewidth=4,
            solid_capstyle='round',
            zorder=14
        )
        
        # 방향 화살표 (더 긴 것)
        arrow_length = length * 0.8
        arrow_end_x = x + sin_y * arrow_length
        arrow_end_z = z + cos_y * arrow_length
        
        ax.annotate(
            '',
            xy=(arrow_end_x, arrow_end_z),
            xytext=(x, z),
            arrowprops=dict(
                arrowstyle='-|>',
                color='yellow',
                lw=2,
                mutation_scale=15
            ),
            zorder=15
        )
    
    def _draw_lidar(self, ax, env):
        """라이다 레이 시각화"""
        if not hasattr(env, 'state') or env.state is None:
            return
        
        x, z, yaw = env.state.x, env.state.z, env.state.yaw
        
        # 라이다 파라미터
        num_rays = getattr(env.config, 'lidar_num_rays', 16) if hasattr(env, 'config') else 16
        max_range = getattr(env.config, 'lidar_max_range', 30) if hasattr(env, 'config') else 30
        
        # 라이다 값 가져오기 (캐스팅 수행)
        if hasattr(env, '_cast_lidar_rays'):
            rays = env._cast_lidar_rays()
        else:
            rays = np.full(num_rays, max_range)
        
        # 각 레이 그리기
        for i, ray_dist in enumerate(rays):
            angle_offset = (i / num_rays) * 360 - 180
            ray_angle = yaw + angle_offset
            ray_angle_rad = math.radians(ray_angle)
            
            # 레이 끝점
            end_x = x + math.sin(ray_angle_rad) * ray_dist
            end_z = z + math.cos(ray_angle_rad) * ray_dist
            
            # 거리에 따른 색상
            ratio = ray_dist / max_range
            if ratio < 0.3:
                color = 'red'
                alpha = 0.8
            elif ratio < 0.6:
                color = 'orange'
                alpha = 0.6
            else:
                color = 'green'
                alpha = 0.4
            
            # 레이 선
            ax.plot(
                [x, end_x], [z, end_z],
                color=color,
                linewidth=1,
                alpha=alpha,
                zorder=6
            )
            
            # 충돌 지점 표시 (max_range보다 작을 때)
            if ray_dist < max_range - 0.5:
                ax.plot(end_x, end_z, 'o', color=color, markersize=3, zorder=6)
        
        # 라이다 범위 (부채꼴) - 선택적
        # wedge = Wedge(
        #     (x, z), max_range, yaw - 180, yaw + 180,
        #     facecolor='blue', alpha=0.05, zorder=1
        # )
        # ax.add_patch(wedge)
    
    def _draw_info_panel(self, ax, env):
        """정보 패널"""
        if not hasattr(env, 'state') or env.state is None:
            return
        
        state = env.state
        
        # 정보 텍스트
        info_lines = [
            f"Position: ({state.x:.1f}, {state.z:.1f})",
            f"Heading: {state.yaw:.1f}°",
            f"Speed: {state.speed:.2f} m/s",
        ]
        
        if hasattr(env, 'goal') and env.goal:
            dist = math.hypot(env.goal[0] - state.x, env.goal[1] - state.z)
            info_lines.append(f"Goal Distance: {dist:.1f} m")
        
        if hasattr(env, '_heading_error'):
            info_lines.append(f"Heading Error: {env._heading_error():.1f}°")
        
        info_lines.extend([
            f"Step: {self.step_in_episode}",
            f"Episode: {self.episode_count}",
            f"Reward: {self.current_episode_reward:.1f}",
        ])
        
        info_text = '\n'.join(info_lines)
        
        # 텍스트 박스
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9)
        ax.text(
            0.02, 0.98, info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=props,
            zorder=20
        )


class LiveVisualizationCallback(VisualizationCallback):
    """
    실시간 시각화 콜백 (matplotlib interactive)
    
    학습 중 실시간으로 창에 표시
    """
    
    def __init__(self, update_freq: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.update_freq = update_freq
        self.fig = None
        self.ax = None
        
    def _on_training_start(self):
        super()._on_training_start()
        plt.ion()  # Interactive mode
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        
    def _on_step(self) -> bool:
        result = super()._on_step()
        
        # 주기적 업데이트
        if self.num_timesteps % self.update_freq == 0:
            self._update_live_viz()
        
        return result
    
    def _update_live_viz(self):
        """실시간 시각화 업데이트"""
        try:
            env = self.training_env.envs[0]
            if hasattr(env, 'env'):
                env = env.env
            
            self.ax.clear()
            
            # 간단한 시각화 (성능 위해)
            if self.show_obstacles and hasattr(env, 'obstacle_rects'):
                for obs in env.obstacle_rects[:100]:  # 최대 100개만
                    if isinstance(obs, tuple):
                        x_min, x_max, z_min, z_max = obs
                    else:
                        x_min, x_max = obs['x_min'], obs['x_max']
                        z_min, z_max = obs['z_min'], obs['z_max']
                    rect = patches.Rectangle(
                        (x_min, z_min), x_max - x_min, z_max - z_min,
                        facecolor='gray', alpha=0.5
                    )
                    self.ax.add_patch(rect)
            
            # 전차
            if hasattr(env, 'state') and env.state:
                self._draw_tank(self.ax, env.state.x, env.state.z, env.state.yaw)
            
            # 목표
            if hasattr(env, 'goal') and env.goal:
                self.ax.plot(env.goal[0], env.goal[1], 'r*', markersize=15)
            
            self.ax.set_xlim(0, 300)
            self.ax.set_ylim(0, 300)
            self.ax.set_aspect('equal')
            self.ax.set_title(f"Step: {self.num_timesteps:,} | Episode: {self.episode_count}")
            
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            pass  # 실시간 시각화 실패는 무시
    
    def _on_training_end(self):
        plt.ioff()
        plt.close(self.fig)


def create_episode_gif(
    image_folder: str,
    output_path: str,
    fps: int = 10,
    pattern: str = "step_*.png"
):
    """에피소드 이미지들을 GIF로 변환"""
    try:
        import imageio
        import glob
        
        images = sorted(glob.glob(os.path.join(image_folder, pattern)))
        
        if not images:
            print(f"⚠️ 이미지를 찾을 수 없습니다: {image_folder}/{pattern}")
            return
        
        frames = [imageio.imread(img) for img in images]
        imageio.mimsave(output_path, frames, fps=fps)
        
        print(f"✅ GIF 생성 완료: {output_path} ({len(frames)} frames)")
        
    except ImportError:
        print("⚠️ imageio가 필요합니다: pip install imageio")
    except Exception as e:
        print(f"❌ GIF 생성 실패: {e}")