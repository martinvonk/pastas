#%%
from typing import List, Protocol
from pandas import DataFrame, Series
import pandas as pd
import matplotlib.pyplot as plt
from numpy import zeros

from pastas.typing import ArrayLike

import flopy



class Modflow(Protocol):
    def __init__(self) -> None:
        ...

    def get_init_parameters(self) -> DataFrame:
        ...

    def create_model(self) -> None:
        ...

    def simulate(self) -> ArrayLike:
        ...

#%%
class ModflowRch:

    def __init__(self, stress: List[Series], initialhead: float, exe_name: str, sim_ws: str):
        self.stress = stress
        self.initialhead = initialhead
        self.exe_name = exe_name
        self.sim_ws = sim_ws
        self._name = "mf_rch"
        self._model = None

    def get_init_parameters(self) -> DataFrame:
        parameters = DataFrame(columns=["initial", "pmin", "pmax", "vary", "name"])
        parameters.loc[self._name + "_sy"] = (0.1, 0.001, 0.5, True, self._name)
        parameters.loc[self._name + "_c"] = (0.001, 0.00001, 0.1, True, self._name)
        parameters.loc[self._name + "_d"] = (self.initialhead, self.initialhead - 10, self.initialhead + 10, True, self._name)
        parameters.loc[self._name + "_f"] = (-1.0, -2.0, 0.0, True, self._name)
        return parameters

    def create_model(
        self, p: ArrayLike,

    ) -> None:

        sy, c, d, f = p[0:4]

        ss = 1.0e-5
        laytyp = 1

        r = self.stress[0] - f * self.stress[1]

        nper = len(r)

        sim = flopy.mf6.MFSimulation(
            sim_name=self._name,
            version="mf6",
            exe_name=self.exe_name,
            sim_ws=self.sim_ws,
        )

        tdis_rc = [(1, 1, 1) for _ in range(nper)]
        _ = flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=nper, perioddata=tdis_rc)

        gwf = flopy.mf6.ModflowGwf(
            sim,
            modelname=self._name,
            newtonoptions="NEWTON",
            save_flows=True,
        )

        imsgwf = flopy.mf6.ModflowIms(
            sim,
            # print_option="SUMMARY",
            complexity="MODERATE",
            # outer_dvclose=1e-9,
            # outer_maximum=100,
            # under_relaxation="DBD",
            # under_relaxation_theta=0.7,
            # inner_maximum=300,
            # inner_dvclose=1e-9,
            # rcloserecord=1e-3,
            # linear_acceleration="BICGSTAB",
            # scaling_method="NONE",
            # reordering_method="NONE",
            # relaxation_factor=0.97,
            # filename=f"{name}.ims",
        )
        sim.register_ims_package(imsgwf, [self._name])

        _ = flopy.mf6.ModflowGwfdis(
            gwf,
            length_units="METERS",
            nlay=1,
            nrow=1,
            ncol=1,
            delr=1,
            delc=1,
            top=self.initialhead + 10,
            botm=self.initialhead - 10,
            idomain=1,
        )

        _ = flopy.mf6.ModflowGwfic(gwf, strt=d)

        _ = flopy.mf6.ModflowGwfnpf(gwf, save_flows=True, icelltype=laytyp)

        _ = flopy.mf6.ModflowGwfsto(
            gwf,
            save_flows=False,
            iconvert=laytyp,
            ss=ss,
            sy=sy,
            transient=True,
        )

        # ghb
        ghbspdict = {0: [[(0, 0, 0), d, c]]}
        _ = flopy.mf6.ModflowGwfghb(
            gwf,
            stress_period_data=ghbspdict,
        )

        _ = flopy.mf6.ModflowGwfoc(
            gwf,
            head_filerecord=f"{gwf.name}.hds",
            saverecord=[("HEAD", "ALL")],
        )

        rch_spd = {i: [[(0, 0, 0), r[i]]] for i in range(nper)}
        _ = flopy.mf6.ModflowGwfrch(gwf, stress_period_data=rch_spd, save_flows=True)

        self._model = sim

    def simulate(self, p: ArrayLike) -> ArrayLike:
        self.create_model(p=p)
        self._model.write_simulation(silent=True)
        success, _ = self._model.run_simulation(silent=False)
        if success:
            heads = flopy.utils.HeadFile(self._model.sim_path / f"{self._model.name}.hds").get_ts(
                (0, 0, 0)
            )
            return heads[:, 1]
        return zeros(prec.shape)

