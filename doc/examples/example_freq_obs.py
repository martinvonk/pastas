"""
This test file is meant for developing purposes. Providing an easy method to
test the functioning of Pastas during development.

"""

import pandas as pd

import pastas as ps

ps.set_log_level("INFO")

# read observations and create the time series model
obs = pd.read_csv("data/head_nb1.csv", index_col=0, parse_dates=True).squeeze("columns")

# Create the time series model
ml = ps.Model(obs, name="head")
ml.add_noisemodel(ps.NoiseModel())

# read weather data
rain = pd.read_csv("data/rain_nb1.csv", index_col=0, parse_dates=True).squeeze(
    "columns"
)
evap = pd.read_csv("data/evap_nb1.csv", index_col=0, parse_dates=True).squeeze(
    "columns"
)

# Create stress
sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Exponential(), name="recharge")
ml.add_stressmodel(sm)

# Solve and only fit on one observation per year
ml.solve(freq_obs="365D")
ml.plot()
