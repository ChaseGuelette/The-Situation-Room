import numpy as np
from collections import defaultdict

class EmotionMatcher:
    def __init__(self, top_n=5):
        self.top_n = top_n
        self.user_emotions = []
        self.ai_emotions = []
        self.temp_ai_emotions = []
        self.emotion_matches = []
        self.success_score = 0
        self.connection_ranges = {
            (0.0, 0.2): -1,
            (0.2, 0.5): 0,
            (0.5, 0.8): 1,
            (0.8, 1.0): 2
        }

    def add_user_emotion(self, emotion_scores):
        user_emotions = self._get_top_n_emotions(emotion_scores)
        self.user_emotions.append(user_emotions)
        
        if self.temp_ai_emotions:
            match_score, best_ai_emotions = self._calculate_best_match(user_emotions, self.temp_ai_emotions)
            self.emotion_matches.append(match_score)
            self.ai_emotions.extend(self.temp_ai_emotions)
            self.temp_ai_emotions = []
            
            success_increment, range_info = self._calculate_success_increment(match_score)
            self.success_score += success_increment
            
            return match_score, user_emotions, best_ai_emotions, success_increment, range_info
        
        return 0, user_emotions, None, 0, "No AI emotions to compare"

    def add_ai_emotions(self, emotion_scores_list):
        ai_emotions = self._get_top_n_emotions(emotion_scores_list)
        self.temp_ai_emotions.append(ai_emotions)
        return ai_emotions

    def _get_top_n_emotions(self, emotion_scores):
        return sorted(emotion_scores, key=lambda x: x[1], reverse=True)[:self.top_n]

    def _calculate_best_match(self, user_emotions, ai_emotions_list):
        best_match = 0
        best_ai_emotions = None
        for ai_emotions in ai_emotions_list:
            match_score = self._calculate_emotion_match(user_emotions, ai_emotions)
            if match_score > best_match:
                best_match = match_score
                best_ai_emotions = ai_emotions
        return best_match, best_ai_emotions

    def _calculate_emotion_match(self, user_emotions, ai_emotions):
        user_dict = dict(user_emotions)
        ai_dict = dict(ai_emotions)
        
        matching_emotions = set(user_dict.keys()) & set(ai_dict.keys())
        if not matching_emotions:
            return 0  # No matching emotions
        
        intensity_ratios = []
        for emotion in matching_emotions:
            user_intensity = user_dict[emotion]
            ai_intensity = ai_dict[emotion]
            ratio = min(user_intensity, ai_intensity) / max(user_intensity, ai_intensity)
            intensity_ratios.append(ratio)
        
        # Calculate the median of intensity ratios
        if len(intensity_ratios) % 2 == 0:
            return sum(sorted(intensity_ratios)[len(intensity_ratios)//2-1:len(intensity_ratios)//2+1]) / 2
        else:
            return sorted(intensity_ratios)[len(intensity_ratios)//2]

    def _calculate_success_increment(self, match_score):
        for (lower, upper), increment in self.connection_ranges.items():
            if lower <= match_score < upper:
                return increment, f"{lower:.1f} - {upper:.1f}"
        return 0, "Out of range"

    def get_average_match(self):
        if not self.emotion_matches:
            return 0
        return np.mean(self.emotion_matches)

    def get_latest_match(self):
        if not self.emotion_matches:
            return 0
        return self.emotion_matches[-1]

    def get_success_score(self):
        return self.success_score

    def reset(self):
        self.user_emotions = []
        self.ai_emotions = []
        self.temp_ai_emotions = []
        self.emotion_matches = []
        self.success_score = 0