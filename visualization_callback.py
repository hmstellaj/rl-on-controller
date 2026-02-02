import os
import math
import numpy as np
import pygame
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional, List, Tuple, Dict, Any

class PygameVisualizationCallback(BaseCallback):
    """
    Pygame 기반 시각화 콜백 (실시간 뷰 + 파일 저장)
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
        
        # Pygame 초기화
        if self.headless:
            # 창 없이 실행 (서버용)
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        else:
            # 실시간 창 띄우기 (로컬용)
            if "SDL_VIDEODRIVER" in os.environ:
                del os.environ["SDL_VIDEODRIVER"]
            
        pygame.init()
        
        if self.headless:
            # 오프스크린 서피스 (화면에 안보임)
            self.surface = pygame.Surface((window_size, window_size))
            self.screen = None
        else:
            # 실제 윈도우 창 생성
            self.screen = pygame.display.set_mode((window_size, window_size))
            self.surface = pygame.Surface((window_size, window_size)) # 더블 버퍼링용
            pygame.display.set_caption("Tank RL Training Live View")
            
        self.font = pygame.font.SysFont("Arial", 16)
        
        # 상태 추적
        self.episode_count = 0
        self.current_episode_reward = 0
        
        # 저장 경로 생성
        os.makedirs(os.path.join(save_path, "episodes"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "steps"), exist_ok=True)

    def _on_step(self) -> bool:
        # 보상 추적
        if self.locals.get('rewards') is not None:
            self.current_episode_reward += self.locals['rewards'][0]
            
        # 1. 실시간 렌더링 (매 스텝 혹은 특정 주기로 실행)
        # 매 스텝 그리면 학습이 너무 느려질 수 있으므로 5스텝마다 갱신
        if not self.headless and self.num_timesteps % 5 == 0:
             self._update_live_view()

        # 2. 파일 저장 로직 (기존과 동일)
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
        
        # 윈도우 이벤트 처리 (중요: 이거 없으면 창이 응답 없음 뜸)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # 창 닫아도 학습은 계속되게 할지, 종료할지 결정 (여기선 무시)
                pass

        # 서피스에 그림 그리기
        self._render_to_surface(env, title=f"Step: {self.num_timesteps} | Reward: {self.current_episode_reward:.1f}")
        
        # 화면에 복사 및 갱신
        if self.screen:
            self.screen.blit(self.surface, (0, 0))
            pygame.display.flip()

    def _on_episode_end(self, env):
        self.episode_count += 1
        
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