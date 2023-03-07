from pandas import DataFrame, Series
from typing import Protocol
from numpy import zeros

from pastas.typing import ArrayLike


try:
    import flopy
except ModuleNotFoundError as e:
    raise e("Modflow requires 'flopy' to be installed.")


class Modflow(Protocol):
    def __init__(self) -> None:
        ...

    def get_init_parameters(self) -> DataFrame:
        ...

    def create_model(self) -> flopy.mf6.MFSimulation:
        ...

    def simulate(self) -> ArrayLike:
        ...

    def get_recharge(self) -> ArrayLike:
        ...


class ModflowRch:
    def __init__(self, initialhead: float, exe_name: str, sim_ws: str):
        self.initialhead = initialhead
        self.exe_name = exe_name
        self.sim_ws = sim_ws

    def get_init_parameters(self, name: str = "mf_rch") -> DataFrame:
        parameters = DataFrame(columns=["initial", "pmin", "pmax", "vary", "name"])
        parameters.loc[name + "_sy"] = (0.1, 0.001, 0.5, True, name)
        parameters.loc[name + "_c"] = (0.001, 0.00001, 0.1, True, name)
        dmean = self.initialhead
        parameters.loc[name + "_d"] = (dmean, dmean - 10, dmean + 10, True, name)
        parameters.loc[name + "_f"] = (-1.0, -2.0, 0.0, True, name)
        return parameters

    def create_model(
        self, prec: ArrayLike, evap: ArrayLike, p: ArrayLike
    ) -> flopy.mf6.MFSimulation:
        sy, c, d, f = p[0:4]

        ss = 1.0e-5
        laytyp = 1

        r = prec - f * evap

        nper = len(r)
        name = "mf_rch"

        sim = flopy.mf6.MFSimulation(
            sim_name=name,
            version="mf6",
            exe_name=self.exe_name,
            sim_ws=self.sim_ws,
        )

        tdis_rc = [(1, 1, 1) for _ in range(nper)]
        _ = flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=nper, perioddata=tdis_rc)

        gwf = flopy.mf6.ModflowGwf(
            sim,
            modelname=name,
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
        sim.register_ims_package(imsgwf, [name])

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

        rch_spd = [[(0, 0, 0), r[i]] for i in range(nper)]
        _ = flopy.mf6.ModflowGwfrch(gwf, stress_period_data=rch_spd, save_flows=True)

        return sim

    def simulate(self, prec: Series, evap: Series, p: ArrayLike) -> ArrayLike:
        sim = self.create_model(prec.values, evap.values, p)
        sim.write_simulation(silent=True)
        success, _ = sim.run_simulation(silent=False)
        if success:
            heads = flopy.utils.HeadFile(f"{self.sim_ws}/{sim.name}.hds").get_ts(
                (0, 0, 0)
            )
            return heads[:, 1]
        return zeros(prec.shape)

    def get_recharge(self, prec: Series, evap: Series, p: ArrayLike) -> ArrayLike:
        return prec.values - p[-1] * evap.values
