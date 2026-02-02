import os
import math
import numpy as np
import pygame
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional, List, Tuple, Dict, Any

class PygameVisualizationCallback(BaseCallback):
    """
    Pygame 기반 시각화 콜백
    - 기본으로 백그라운드 실행
    - 지정된 주기마다 팝업 창 띄워서 에피소드 관전
    """
    
    def __init__(
        self,
        save_path: str = "./viz",
        save_freq: int = 5000,
        episode_save_freq: int = 50,
        map_size: float = 300.0,
        window_size: int = 800,
        show_lidar: bool = True,
        show_path: bool = True,
        headless: bool = False,   # <--- False로 설정하면 창이 뜹니다!
        render_freq: int = 0,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self.episode_save_freq = episode_save_freq
        self.map_size = map_size
        self.window_size = window_size
        self.scale = window_size / map_size
        
        self.show_lidar = show_lidar
        self.show_path = show_path
        self.headless = headless
        self.render_freq = render_freq

        # 윈도우 상태 관리
        self.is_view_open = False
        self.view_episode_count = 0
        self.surface = pygame.Surface((window_size, window_size))
        self.screen = None
        self.font = None

        # Pygame 초기화
        pygame.init()
        self.font = pygame.font.SysFont("Arial", 16)

        # render_freq 기반으로 창 띄우기
        if not self.headless and self.render_freq == 0:
            self._open_window()
        elif self.headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            
        # 상태 추적
        self.episode_count = 0
        self.current_episode_reward = 0
        
        # 저장 경로 생성
        os.makedirs(os.path.join(save_path, "episodes"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "steps"), exist_ok=True)

    def _open_window(self):
        if not self.is_view_open and not self.headless:
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption(f"Tank RL Training Live View (Step {self.num_timesteps})")
            self.is_view_open = True
            self.view_episode_count = 0
            print("Start Live View")

    def _close_window(self):
        if self.is_view_open:
            pygame.display.quit()
            self.screen = None
            self.is_view_open = False
            print("Finish Live View")

    def _on_step(self) -> bool:
        # 보상 추적
        if self.locals.get('rewards') is not None:
            self.current_episode_reward += self.locals['rewards'][0]

        # 스크린 팝업 트리거 체크
        if not self.headless and self.render_freq > 0:
            if self.num_timesteps > 0 and self.num_timesteps % self.render_freq == 0:
                self._open_window()
        
        # 창이 열려있을때 화면 갱신
        if self.is_view_open:
            self._update_live_view()

        # 파일 저장 로직
        is_save_step = self.num_timesteps % self.save_freq == 0
        is_episode_end = self.locals.get('dones') is not None and self.locals['dones'][0]
        
        if is_save_step or is_episode_end:
            env = self._get_env()
            
            if is_save_step:
                filename = os.path.join(self.save_path, "steps", f"step_{self.num_timesteps:08d}.png")
                self._render_to_surface(env, title=f"Step {self.num_timesteps}")
                self._save_surface(filename)
            
            if is_episode_end:
                self._on_episode_end(env)
                
        return True

    def _get_env(self):
        if hasattr(self.training_env, 'envs'):
            return self.training_env.envs[0]
        else:
            return self.training_env.get_attr('env', indices=0)[0]

    def _update_live_view(self):
        """실시간 화면 갱신"""
        env = self._get_env()
        
        # 윈도우 이벤트 처리 (응답 없음 방지)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._close_window()
        
        env = self._get_env()
        self._render_to_surface(env, title=f"Step: {self.num_timesteps} | Reward: {self.current_episode_reward:.1f}")
        
        # 화면에 복사 및 갱신
        if self.screen:
            self.screen.blit(self.surface, (0, 0))
            pygame.display.flip()

    def _on_episode_end(self, env):
        self.episode_count += 1

        # 팝업 모드일때 에피소드가 끝나면 창 닫기
        if self.is_view_open and self.render_freq > 0:
            self.view_episode_count += 1
            if self.view_episode_count >= 5:
                self._close_window()
        
        if self.episode_count % self.episode_save_freq == 0:
            info = self.locals.get('infos', [{}])[0]
            result = "GOAL" if info.get('reached_goal') else ("CRASH" if info.get('collision') else "TIME")
            filename = os.path.join(self.save_path, "episodes", f"ep_{self.episode_count:05d}_{result}.png")
            
            self._render_to_surface(env, title=f"Ep {self.episode_count} {result} (R={self.current_episode_reward:.0f})")
            self._save_surface(filename)
            
        self.current_episode_reward = 0

    def _coord(self, x, z):
        px = int(x * self.scale)
        py = int(self.window_size - (z * self.scale))
        return px, py

    def _save_surface(self, filename):
        pygame.image.save(self.surface, filename)

    def _render_to_surface(self, env, title=""):
        """서피스에 현재 상태 그리기 (화면 표시/저장 공통)"""
        # 1. 배경
        self.surface.fill((240, 240, 240))
        
        # 2. 장애물
        if hasattr(env, 'obstacle_rects'):
            for obs in env.obstacle_rects:
                if isinstance(obs, tuple):
                    xmin, xmax, zmin, zmax = obs
                else:
                    xmin, xmax = obs['x_min'], obs['x_max'], obs['z_min'], obs['z_max']
                
                x, y = self._coord(xmin, zmax)
                w = (xmax - xmin) * self.scale
                h = (zmax - zmin) * self.scale
                pygame.draw.rect(self.surface, (100, 100, 100), (x, y, w, h))

        # 3. 경로
        if self.show_path and hasattr(env, 'path') and env.path:
            points = [self._coord(p[0], p[1]) for p in env.path]
            if len(points) > 1:
                pygame.draw.lines(self.surface, (0, 0, 255), False, points, 2)
            if env.goal:
                gx, gy = self._coord(env.goal[0], env.goal[1])
                pygame.draw.circle(self.surface, (255, 0, 0), (gx, gy), 8)

        # 4. 전차
        if hasattr(env, 'state') and env.state:
            tx, ty = self._coord(env.state.x, env.state.z)
            
            # 본체
            pygame.draw.circle(self.surface, (0, 150, 0), (tx, ty), 6)
            
            # 헤딩 라인
            rad = math.radians(env.state.yaw)
            dx = math.sin(rad) * 15
            dy = math.cos(rad) * 15
            end_x = tx + dx
            end_y = ty - dy
            pygame.draw.line(self.surface, (0, 255, 0), (tx, ty), (end_x, end_y), 3)

            # 5. 라이다
            if self.show_lidar and hasattr(env, '_cast_lidar_rays'):
                rays = env._cast_lidar_rays()
                num_rays = len(rays)
                max_range = env.config.lidar_max_range
                
                for i, dist in enumerate(rays):
                    angle_offset = (i / num_rays) * 360 - 180
                    ray_angle = env.state.yaw + angle_offset
                    ray_rad = math.radians(ray_angle)
                    
                    lx = math.sin(ray_rad) * dist * self.scale
                    ly = math.cos(ray_rad) * dist * self.scale
                    
                    color = (255, 0, 0) if dist < max_range - 0.1 else (0, 255, 0)
                    pygame.draw.line(self.surface, color, (tx, ty), (tx + lx, ty - ly), 1)

        # 6. 정보 텍스트
        text = self.font.render(title, True, (0, 0, 0))
        self.surface.blit(text, (10, 10))
        
    def _on_training_end(self):
        pygame.quit()