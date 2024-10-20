import google.generativeai as genai

class LLMDemandChecker:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    def check_demand(self, demand, transcript):
        prompt = f"""
        Given the following demand in a negotiation:
        "{demand.description}" (Current value: {demand.current_value})

        And the following transcript of the negotiation:
        "{transcript}"

        Has the negotiator successfully met this demand? Please respond with only 'True' if the demand has been met, or 'False' if it has not been met.
        """

        response = self.model.generate_content(prompt)
        return response.text.strip().lower() == 'true'

    def check_all_demands(self, demands, transcript):
        results = []
        for demand in demands:
            is_met = self.check_demand(demand, transcript)
            results.append((demand, is_met))
        return results