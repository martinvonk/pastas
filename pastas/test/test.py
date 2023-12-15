#%%
import pandas as pd
import pastas as ps
import matplotlib.pyplot as plt

from pastas.timer import SolveTimer

%load_ext autoreload
%autoreload 2
#%%
tmin = pd.Timestamp("2001-01-01")
tmax = pd.Timestamp("2014-12-31")
tmin_wu = tmin - pd.Timedelta(days=3651)
tmin_wu = pd.Timestamp("1986-01-01")
head = (
    pd.read_csv("../../doc/examples/data/head_nb1.csv",index_col="date", parse_dates=True)
    .squeeze()
    .loc[tmin:tmax]
)
prec = (
    pd.read_csv("../../doc/examples/data/rain_nb1.csv", index_col="date", parse_dates=True)
    .squeeze()
    .loc[tmin_wu:tmax]
)
evap = (
    pd.read_csv("../../doc/examples/data/evap_nb1.csv", index_col="date", parse_dates=True)
    .squeeze()
    .loc[tmin_wu:tmax]
)

mf6_exe = r"C:\Users\vonkm\Documents\Modflow_USG\bin\mf6.exe"
ps.plots.series(head, [prec, evap], hist=False)
#%%
# create model with exponential response function
mlexp = ps.Model(head, noisemodel=False)
mlexp.add_stressmodel(ps.RechargeModel(prec=prec, evap=evap, rfunc=ps.Exponential(), name="test_exp"))
mlexp.solve(tmin=tmin, tmax=tmax, noise=False)
mlexp.plot()

#%%
# extract resistance and sy from exponential model
mlexp_c = mlexp.parameters.loc["test_exp_A", "optimal"]
mlexp_c_i = mlexp.parameters.loc["test_exp_A", "initial"]
mlexp_sy = mlexp.parameters.loc["test_exp_a", "optimal"] / mlexp.parameters.loc["test_exp_A", "optimal"]
mlexp_sy_i = mlexp.parameters.loc["test_exp_a", "initial"] / mlexp.parameters.loc["test_exp_A", "initial"]
mlexp_d = mlexp.parameters.loc["constant_d", "optimal"]
mlexp_d_i = mlexp.parameters.loc["constant_d", "initial"]
mlexp_f = mlexp.parameters.loc["test_exp_f", "optimal"]
mlexp_f_i = mlexp.parameters.loc["test_exp_f", "initial"]


#%%
# create modflow pastas model with c and sy
mlexpmf = ps.Model(head, noisemodel=False)
expmf = ps.ModflowRch(exe_name=mf6_exe, sim_ws="test_expmf")
expsm = ps.ModflowModel([prec, evap], modflow=expmf, name="test_expmfsm")
mlexpmf.add_stressmodel(expsm)
mlexpmf.set_parameter(f"{expsm.name}_sy", initial=mlexp_sy, vary=False)
mlexpmf.set_parameter(f"{expsm.name}_c", initial=mlexp_c, vary=False)
mlexpmf.set_parameter(f"{expsm.name}_f", initial=mlexp_f, vary=False)
mlexpmf.set_parameter("constant_d", initial=mlexp_d, vary=False)
# mlexpmf.solve(noise=False)
mlexpmf.plot()
#%%
ml = ps.Model(head, constant=True, noisemodel=False)
mf = ps.ModflowRch(exe_name=mf6_exe, sim_ws="test_mfrch")
sm = ps.ModflowModel([prec, evap], modflow=mf, name="test_mfsm")
ml.add_stressmodel(sm)
ml.set_parameter(f"{sm.name}_sy", initial=mlexp_sy_i, vary=True)
ml.set_parameter(f"{sm.name}_c", initial=mlexp_c_i, vary=True)
ml.set_parameter(f"{sm.name}_f", initial=mlexp_f_i, vary=True)
ml.set_parameter("constant_d", initial=mlexp_d_i, vary=True)
solver = ps.LeastSquares(method="lm")
with SolveTimer() as st:
    ml.solve(noise=False, solver=solver, callback=st.timer, report=False)
ml.fit.result
#%%
ml.plot()

#%%
ml.parameters
#%%
mlexp.parameters

#%%
ax = ml.plot()
mlexp.plot(ax=ax)
#%%
ax = mlexpmf.plot()
mlexp.plot(ax=ax)

#%%
