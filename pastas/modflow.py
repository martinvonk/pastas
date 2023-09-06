from typing import List, Protocol

import flopy
import numpy as np
from pandas import DataFrame, Series

from pastas.typing import ArrayLike


class Modflow(Protocol):
    def __init__(self) -> None:
        ...

    def get_init_parameters(self) -> DataFrame:
        ...

    def create_model(self) -> None:
        ...

    def simulate(self) -> ArrayLike:
        ...


class ModflowRch:
    def __init__(
        self, stress: List[Series], initialhead: float, exe_name: str, sim_ws: str
    ):
        self.stress = stress
        self.initialhead = initialhead
        self.exe_name = exe_name
        self.sim_ws = sim_ws
        self._nper = len(stress[0])
        self._name = "mf_rch"
        self._simulation = None
        self._gwf = None
        self.create_model()

    def get_init_parameters(self) -> DataFrame:
        parameters = DataFrame(columns=["initial", "pmin", "pmax", "vary", "name"])
        parameters.loc[self._name + "_sy"] = (0.1, 0.001, 0.5, True, self._name)
        parameters.loc[self._name + "_c"] = (0.001, 0.00001, 0.1, True, self._name)
        parameters.loc[self._name + "_d"] = (
            self.initialhead,
            self.initialhead - 10,
            self.initialhead + 10,
            True,
            self._name,
        )
        parameters.loc[self._name + "_f"] = (-1.0, -2.0, 0.0, True, self._name)
        return parameters

    def create_model(self) -> None:
        sim = flopy.mf6.MFSimulation(
            sim_name=self._name,
            version="mf6",
            exe_name=self.exe_name,
            sim_ws=self.sim_ws,
        )

        _ = flopy.mf6.ModflowTdis(
            sim,
            time_units="DAYS",
            nper=self._nper,
            perioddata=[(1, 1, 1) for _ in range(self._nper)],
        )

        gwf = flopy.mf6.ModflowGwf(
            sim,
            modelname=self._name,
            newtonoptions="NEWTON",
        )

        imsgwf = flopy.mf6.ModflowIms(
            sim,
            # print_option="SUMMARY",
            complexity="SIMPLE",
            # outer_dvclose=1e-9,
            # outer_maximum=100,
            # under_relaxation="DBD",
            # under_relaxation_theta=0.7,
            # inner_maximum=300,
            # inner_dvclose=1e-9,
            # rcloserecord=1e-3,
            linear_acceleration="BICGSTAB",
            # scaling_method="NONE",
            # reordering_method="NONE",
            # relaxation_factor=0.97,
            # filename=f"{name}.ims",
            pname=None,
        )
        # sim.register_ims_package(imsgwf, [self._name])

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
            pname=None,
        )

        _ = flopy.mf6.ModflowGwfoc(
            gwf,
            head_filerecord=f"{gwf.name}.hds",
            saverecord=[("HEAD", "ALL")],
            pname=None,
        )

        sim.write_simulation(silent=True)
        self._simulation = sim
        self._gwf = gwf

    def update_model(self, p: ArrayLike):
        sy, c, d, f = p[0:4]

        ss = 1.0e-5
        laytyp = 1

        r = self.stress[0] - f * self.stress[1]

        ic = flopy.mf6.ModflowGwfic(self._gwf, strt=d)
        ic.write()

        npf = flopy.mf6.ModflowGwfnpf(self._gwf, save_flows=False, icelltype=laytyp)
        npf.write()

        sto = flopy.mf6.ModflowGwfsto(
            self._gwf,
            save_flows=False,
            iconvert=laytyp,
            ss=ss,
            sy=sy,
            transient=True,
        )
        sto.write()

        # ghb
        ghb = flopy.mf6.ModflowGwfghb(
            self._gwf,
            maxbound=1,
            stress_period_data={0: [[(0, 0, 0), d, c]]},
            pname="ghb",
        )
        ghb.write()

        rch = flopy.mf6.ModflowGwfrch(
            self._gwf,
            maxbound=1,
            pname="rch",
            stress_period_data={0: [[(0, 0, 0), "recharge"]]},
        )
        rts = [(i, x) for i, x in zip(range(self._nper + 1), np.append(r, 0.0))]
        rch.ts.initialize(
            filename="recharge.ts",
            timeseries=rts,
            time_series_namerecord="recharge",
            interpolation_methodrecord="stepwise",
            sfacrecord=1.1,
            pname="rchts",
        )
        rch.write()
        rch.ts.write()

        self._gwf.name_file.write()

    def simulate(self, p: ArrayLike) -> ArrayLike:
        self.update_model(p=p)
        # self._simulation.write_simulation(silent=True)
        success, _ = self._simulation.run_simulation(silent=False)
        if success:
            heads = flopy.utils.HeadFile(
                self._simulation.sim_path / f"{self._simulation.name}.hds"
            ).get_ts((0, 0, 0))
            return heads[:, 1]
        return np.zeros(self.stress[0].shape)
