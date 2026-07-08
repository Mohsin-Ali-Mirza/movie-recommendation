from locust import HttpUser, task, between

class MovieRecommendationUser(HttpUser):
    wait_time = between(1, 3)  # Simulates real-world user wait time

    @task
    def test_root_api(self):
        self.client.get("/")  # Hits the root API endpoint

    # @task
    # def test_recommendation_api(self):
    #     payload = {
    #         "Ironman": 4,
    #         "Spider-man": 4,
    #         "Avengers:": 4,
    #     }
    #     headers = {"Content-Type": "application/json"}
    #     self.client.post("/recommend_movies_by_content_based", json=payload, headers=headers)

