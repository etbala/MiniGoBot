

class EloEvaluator:
    def __init__(self, initial_rating=1200):
        self.ratings = {}  # Dictionary to store Elo ratings
        self.initial_rating = initial_rating

    def add_model(self, model_name):
        if model_name not in self.ratings:
            self.ratings[model_name] = self.initial_rating

    def update_ratings(self, model_a, model_b, result_a):
        """
        Update Elo ratings for two models after a game.
        Args:
            model_a (str): Name of model A.
            model_b (str): Name of model B.
            result_a (float): Result for model A (1 = win, 0 = loss, 0.5 = draw).
        """
        R_A = self.ratings[model_a]
        R_B = self.ratings[model_b]

        # Calculate expected scores
        E_A = 1 / (1 + 10 ** ((R_B - R_A) / 400))
        E_B = 1 - E_A

        # Update ratings
        K = 32  # Scaling factor
        self.ratings[model_a] += K * (result_a - E_A)
        self.ratings[model_b] += K * ((1 - result_a) - E_B)

    def get_rating(self, model_name):
        return self.ratings.get(model_name, self.initial_rating)

    def print_ratings(self):
        for model, rating in sorted(self.ratings.items(), key=lambda x: -x[1]):
            print(f"{model}: {rating:.2f}")
