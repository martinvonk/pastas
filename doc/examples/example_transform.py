import pastas as ps

# load a model with precipitation, evaporation and a well
ml = ps.io.load("data/B28H1808_2.pas")

# first solve and plot without a transform to see the bad model-performance
ml.solve()
ax = ml.plots.decomposition()

# then solve and plot with a ThresholdTransform
ml.add_transform(ps.ThresholdTransform())
ml.solve()
ax = ml.plots.decomposition()
