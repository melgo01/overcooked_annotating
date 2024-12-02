import os
import csv
import json
import pygame
import argparse
import numpy as np
from overcooked_ai.overcooked_ai_py.env import OverCookedEnv_Play

PROGRESS_FILE = "progress.json"

# JSON 파일 읽기
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    
def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as file:
        json.dump(progress, file)

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as file:
            return json.load(file)
    return {}
    
# Player 1과 Player 2의 행동을 분리
def split_player_actions(ep_actions):
    player_1_actions = []
    player_2_actions = []
    
    for action_pair in ep_actions[0]:
        player_1_actions.append(action_pair[0])
        player_2_actions.append(action_pair[1])
    
    return player_1_actions, player_2_actions

# Define the action-to-index mapping
ACTION_TO_INDEX = { 
    (0, -1): 0,  # NORTH
    (0, 1): 1,   # SOUTH
    (1, 0): 2,   # EAST
    (-1, 0): 3,  # WEST
    (0, 0): 4,   # STAY
    'INTERACT': 5  # INTERACT
}

def convert_actions_to_indices(action_list):
    """
    Convert action list from tuple/string format to integer indices.

    Args:
        action_list (list): List of actions (e.g., [(0, -1), (1, 0), 'interact'])

    Returns:
        list: List of actions converted to indices
    """
    # return [ACTION_TO_INDEX[action] for action in action_list]
    return [ACTION_TO_INDEX[tuple(action) if isinstance(action, list) else action] for action in action_list]


# 시각화를 위한 Workspace 클래스
class Workspace:
    def __init__(self, args, json_path):
        self.args = args
        self.json_path = json_path

    def run(self):
        # Load JSON file and extract actions
        data = load_json(self.json_path)
        ep_actions = data["ep_actions"]
        p0_action_list, p1_action_list = split_player_actions(ep_actions)
        p0_action_list = convert_actions_to_indices(p0_action_list)
        p1_action_list = convert_actions_to_indices(p1_action_list)

        # Extract layout_name and set as scenario
        layout_name = data["mdp_params"][0].get("layout_name", "default_scenario")
        self.args.scenario = layout_name
        print(f"Scenario set for OverCookedEnv_Play: {layout_name}")

        # Initialize environment
        env = OverCookedEnv_Play(scenario=self.args.scenario, episode_length=self.args.epi_length)
        obs = env.reset()

        clock = pygame.time.Clock()
        try:
            # Initialize Pygame screen
            image = env.render()
            screen = pygame.display.set_mode((image.shape[1], image.shape[0]))
            pygame.display.set_caption(f"Visualization: {os.path.basename(self.json_path)}")
            screen.blit(pygame.surfarray.make_surface(np.rot90(np.flip(image[..., ::-1], 1))), (0, 0))
            pygame.display.flip()

            for step, (p0_action, p1_action) in enumerate(zip(p0_action_list, p1_action_list)):
                clock.tick(6.67)

                human_action = [p0_action, p1_action]
                obs, _, _, _ = env.step(action=human_action)

                image = env.render(step)
                screen.blit(pygame.surfarray.make_surface(np.rot90(np.flip(image[..., ::-1], 1))), (0, 0))
                pygame.display.flip()

        finally:
            pygame.quit()

def label_sequences(input_csv, output_csv, base_path, args):
    """
    사용자 입력을 통해 CSV 데이터를 라벨링하고 저장.
    """
    progress = load_progress()
    last_index = progress.get("last_index", 0)

    existing_data = []
    if os.path.exists(output_csv):
        with open(output_csv, 'r') as outfile:
            reader = csv.reader(outfile)
            header = next(reader)
            existing_data = list(reader)
        print(f"Loaded {len(existing_data)} rows from existing output CSV.")

    with open(input_csv, 'r') as infile, open(output_csv, 'a', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # header = next(reader)
        # if "HumanLabel" not in header:
        #     header.append("HumanLabel")
        # writer.writerow(header)

        input_header = next(reader)
        if not existing_data:  # output_csv가 비어 있는 경우 헤더 작성
            writer.writerow(input_header + ["HumanLabel"])

        rows = list(reader)

        try:
            for idx, row in enumerate(rows):
                if idx < last_index:
                    continue  # Skip already labeled rows

                data1_path = os.path.join(base_path, f"{row[0]}.json")
                data2_path = os.path.join(base_path, f"{row[1]}.json")

                while True:  # Allow replays
                    print(f"\nVisualizing: {row[0]} and {row[1]}")
                    for json_path in [data1_path, data2_path]:
                        workspace = Workspace(args, json_path)
                        workspace.run()
                        pygame.time.wait(1000)

                    replay = input("Do you want to replay the data? (yes/no): ").strip().lower()
                    if replay == "no":
                        break

                while True:  # Get user label
                    try:
                        label = int(input("Enter your label (0:Data1 Better , 1: Data2 Better, 2: Both Equal): "))
                        if label in {0, 1, 2}:
                            break
                    except ValueError:
                        pass
                    print("Invalid input. Please enter 0, 1, or 2.")

                row.append(label)
                writer.writerow(row)
                print(f"Saved label: {label}")

                # Save progress after each row
                save_progress({"last_index": idx + 1})

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\nProgress saved. Exiting...")
            save_progress({"last_index": idx})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Labeling Tool for Overcooked Data")
    parser.add_argument('--epi_length', type=int, default=400, help='Episode length for visualization')
    parser.add_argument('--json_folder', type=str, default='sliced_jsons/coord', help='Path to the folder containing JSON files')
    parser.add_argument('--input_csv', type=str, default='llm_script_annotations_coord.csv', help='Input CSV file with data pairs')
    parser.add_argument('--output_csv', type=str, default='all_annotations_coord.csv', help='Output CSV file to save labels')
    args = parser.parse_args()

    label_sequences(args.input_csv, args.output_csv, args.json_folder, args)
