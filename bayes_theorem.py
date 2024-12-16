import numpy as np

# Define array of probabilities
probability_init = np.array([0.5, 0.25, 0.25])

# Define weather states (0 = sunny, 1 = rainy, 2 = foggy)
index_of_weather = [0, 1, 2]
state_of_weather = ["Sunny", "Rainy", "Foggy"]


# Define transition probabilities for tomorrow
#               tomorrow
#   today   sunny   rainy   foggy
#   sunny   0.8     0.05    0.15
#   rainy   0.2     0.6     0.2
#   foggy   0.2     0.3     0.5
probability_transition = np.array(
    [
        [0.8, 0.05, 0.15],
        [0.2, 0.6, 0.2],
        [0.2, 0.3, 0.5]
    ]
)

# Define conditional probabilities (going out and observing that someone has an umbrella or not) of evidence given weather condition
#                           sunny   rainy   foggy
# Weather | no umbrella     0.9     0.2     0.7
# Weather | yes umbrella    0.1     0.8     0.3
probability_observation = np.array(
    [
        [0.9, 0.2, 0.7],
        [0.1, 0.8, 0.3]
    ]
)

# If today is sunny what is the most likely forecast for the next two days if you have no umbrella observations to work with?

# FIRST DAY FORECAST
# Define todays state
todays_state = 0
first_day_forecast_array = probability_init[todays_state]
first_day_forecast = np.argmax(first_day_forecast_array)

# SECOND DAY FORECAST
second_day_forecast_array = probability_init[first_day_forecast]
second_day_forecast = np.argmax(second_day_forecast_array)

# print(f"Given that today is a {state_of_weather[todays_state]} day, tomorrow will likely be {state_of_weather[first_day_forecast]} and the day after will be {state_of_weather[second_day_forecast]}")

# Now introduce the conditional probabilities
no_umbrella_weather_probabilities = probability_observation[0]
yes_umbrella_weather_probabilities = probability_observation[1]

# If on the first day, you see no umbrella, what is the probability for each weather state
probability_no_umbrella = np.sum(no_umbrella_weather_probabilities * probability_init)

probability_state_day0_given_umbrella = [
    (no_umbrella_weather_probabilities[0] * probability_init) / probability_no_umbrella,
    (no_umbrella_weather_probabilities[1] * probability_init) / probability_no_umbrella,
    (no_umbrella_weather_probabilities[2] * probability_init) / probability_no_umbrella,
]

sunny_percent = probability_state_day0_given_umbrella[0] * 100
rainy_percent = probability_state_day0_given_umbrella[1] * 100
foggy_percent = probability_state_day0_given_umbrella[2] * 100

array_of_probability = np.array([sunny_percent, rainy_percent, foggy_percent])

most_likely_state_of_weather = state_of_weather[np.argmax(probability_state_day0_given_umbrella)]

for index in enumerate(index_of_weather):
    print(f"The probability of a {index} Day given no umbrella is: {array_of_probability[index]:.2f} %.")
    
# END OF PROJECT





