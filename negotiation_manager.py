class Demand:
    def __init__(self, description, initial_value, feasibility, levels):
        self.description = description
        self.initial_value = initial_value
        self.current_value = initial_value
        self.feasibility = feasibility
        self.levels = levels
        self.current_level = 0
        self.agreed = False

    def adjust_value(self, level):
        if 0 <= level < len(self.levels):
            self.current_level = level
            level_data = self.levels[level]
            self.current_value = level_data['value']
            self.feasibility = level_data['feasibility']

class NegotiationManager:
    def __init__(self, emotion_matcher, scenario_manager, llm_checker):
        self.emotion_matcher = emotion_matcher
        self.scenario_manager = scenario_manager
        self.llm_checker = llm_checker
        self.demands = []
        self.winning_score = 0
        self.demands_met = 0
        self.transcript = ""

    def load_demands(self):
        scenario_demands = self.scenario_manager.get_demands()
        self.demands = [Demand(**demand) for demand in scenario_demands]

    def adjust_demands(self):
        success_score = self.emotion_matcher.get_success_score()
        level = min(max(success_score // 2, 0), len(self.demands[0].levels) - 1)
        for demand in self.demands:
            demand.adjust_value(level)

    def agree_to_demand(self, index):
        if index < 0 or index >= len(self.demands):
            return False, "Invalid demand index"

        demand = self.demands[index]
        if demand.agreed:
            return False, "Demand already agreed to"

        demand.agreed = True
        self.demands_met += 1
        self.emotion_matcher.success_score += 1

        # Adjust winning score based on feasibility
        if demand.feasibility > 0.7:
            self.winning_score += 2
        elif demand.feasibility > 0.4:
            self.winning_score += 1
        else:
            self.winning_score -= 1

        return True, f"Agreed to: {demand.description} (Value: {demand.current_value})"

    def get_current_demands(self):
        return [(i, d.description, d.current_value, d.feasibility, d.agreed) 
                for i, d in enumerate(self.demands)]

    def get_winning_score(self):
        return self.winning_score

    def get_demands_met(self):
        return self.demands_met
    
    def update_transcript(self, message):
        self.transcript += f"\n{message}"

    def check_demands(self):
        met_demands = []
        for demand, is_met in self.llm_checker.check_all_demands(self.demands, self.transcript):
            if is_met and not demand.agreed:
                success, msg = self.agree_to_demand(self.demands.index(demand))
                if success:
                    met_demands.append((self.demands.index(demand), msg))
        return met_demands

    def reset(self):
        for demand in self.demands:
            demand.current_value = demand.initial_value
            demand.current_level = 0
            demand.agreed = False
        self.winning_score = 0
        self.demands_met = 0
        self.transcript = ""

    def adjust_single_demand(self, index, new_level):
        if 0 <= index < len(self.demands):
            self.demands[index].adjust_value(new_level)
            return True
        return False