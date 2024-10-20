import json
import os

class ScenarioManager:
    def __init__(self, scenarios_file='scenarios.json'):
        self.scenarios_file = scenarios_file
        self.scenarios = self.load_scenarios()
        self.current_scenario = None

    def load_scenarios(self):
        if not os.path.exists(self.scenarios_file):
            raise FileNotFoundError(f"Scenarios file '{self.scenarios_file}' not found.")
        
        with open(self.scenarios_file, 'r') as f:
            return json.load(f)

    def get_scenario_names(self):
        return list(self.scenarios.keys())

    def select_scenario(self, scenario_name):
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found.")
        
        self.current_scenario = self.scenarios[scenario_name]
        return self.current_scenario

    def get_current_scenario(self):
        if not self.current_scenario:
            raise ValueError("No scenario has been selected.")
        return self.current_scenario

    def get_demands(self):
        if not self.current_scenario:
            raise ValueError("No scenario has been selected.")
        return self.current_scenario['demands']