import argparse
import os
from overcooked_ai_py.mdp.overcooked_env import OverCookedEnv
import pygame
from collections import deque

class Workspace(object):
    def __init__(self, args):
        self.args = args
    
    def run(self):

        device = 'cpu'

        num_processes = 1
        env = OverCookedEnv(scenario=self.args.scenario, episode_length=self.args.epi_length, time_limits=self.args.time_limits, tutorial=True)
        obs = env.reset()

        clock = pygame.time.Clock()
        try:
            # Wait for starting
            image = env.render()
            screen = pygame.display.set_mode((image.shape[1], image.shape[0]))
            screen.blit(pygame.surfarray.make_surface(np.rot90(np.flip(image[...,::-1],1))), (0,0))
            pygame.display.flip()

            flag = True
            while flag:
                events = pygame.event.get()
                for event in events:
                    if event.type == pygame.KEYDOWN:
                        flag = False

            while curr_state.timestep < args.epi_length:
                clock.tick(6.67)
                
                    
                human_action = [p0_action, p1_action]

                obs, reward, done, info = env.step(action=human_action)
               
                image = env.render(game_time)

                screen.blit(pygame.surfarray.make_surface(np.rot90(np.flip(image[...,::-1],1))), (0,0))
                pygame.display.flip()

        finally:
            pygame.quit()



if __name__ == '__main__':
    # And Play!!!
    parser = argparse.ArgumentParser(description="Example overcooked")
    parser.add_argument('--scenario', type=str, default='simple')
    parser.add_argument('--time_limits', type=int, default=15)
    parser.add_argument('--epi_length', type=int, default=100)
    parser.add_argument('--working_directory', type=str, default='./overcooked_ai/play_overcooked/result')
    args = parser.parse_args()

    os.makedirs(args.working_directory, exist_ok=True)
    workspace = Workspace(args)
    workspace.run()