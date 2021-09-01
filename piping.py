"""Utilities for hydraulics calculations.

Contains functions for hydrodynamic calculations. The main source of the
equations is Crane TP-410.
"""
from math import pi, sin, log, log10, sqrt, tan
from . import logger
from . import ureg, Q_
from .functions import AIR
from .functions import stored_energy
from .functions import Re
from . import os, __location__
from pint import set_application_registry
from serialize import load
from scipy.optimize import root_scalar
from collections.abc import MutableSequence

set_application_registry(ureg)


class HydraulicError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)


class ChokedFlow(HydraulicError):
    def __init__(self, message):
        self.message = message
        super().__init__(message)


def _load_table(table_name):
    yaml_table = load((os.path.join(__location__, table_name)))
    result = {}
    for D, sub_table in yaml_table.items():
        result.update({D: {k: v*ureg.inch for k, v in sub_table.items()}})
    return result


NPS_table = _load_table('NPS_table.yaml')
COPPER_TABLE = _load_table('copper_table.yaml')


class Tube:
    """
    Tube, requires OD and wall thickness specified
    """
    def __init__(self, OD, wall=0*ureg.m, L=0*ureg.m, c=0*ureg.m,
                 eps=0.0018*ureg.inch):
        """Generate tube object.

        Parameters
        ----------
        OD : ureg.Quantity {length: 1}
            Outer diameter of the tube.
        wall : ureg.Quantity {length: 1}
            Wall thickness of the tube.
        L : ureg.Quantity {length: 1}
            Length of the tube.
        c : ureg.Quantity {length: 1}
            Sum of the mechanical allowances plus corrosion and erosion
            allowances.
        eps : ureg.Quantity {length: 1}
            Absolute roughness for the tube. Default value for smooth
            pipe.
        """
        self.OD = OD
        self.D = OD.to(ureg.inch).magnitude
        self.wall = wall
        self.ID = self._calculate_ID()
        self.L = L
        self.area = self._calculate_area()
        self.volume = self._calculate_volume()
        self.eps = eps
        # c = Q_('0.5 mm') for unspecified machined surfaces
        # TODO add calculation for c based on thread depth c = h of B1.20.1
        # Wall thickness under tolerance is 12.5% as per ASTM A999
        wall_tol = 0.125 * self.wall
        self.c = c + wall_tol
        self.type = 'Tube'
        self.S = None
        self.E = None
        self.W = None
        self.Y = None

    def _calculate_ID(self):
        """ureg.Quantity {length: 1} : Wall thickness of Pipe based on NPS table.
        """
        return self.OD - 2*self.wall

    def _calculate_area(self):
        """ureg.Quantity {length: 2} : Cross sectional area of pipe.
        """
        return pi * self.ID**2 / 4

    def _calculate_volume(self):
        """ureg.Quantity {length: 3} : Pipe inner volume.
        """
        return self.area * self.L

    def K(self, Re_):
        """ureg.Quantity {length: 1}: Resistance coefficient.
        """
        eps_r = self.eps / self.ID
        return f_Darcy(Re_, eps_r)*self.L/self.ID

    def f_T(self):
        """Calculate Darcy friction factor for complete turbulence for smooth
        pipe.

        Returns
        -------
        ureg.Quantity {dimensionless}
            Fully turbulent Darcy friction factor.
        """
        return 1 / (4*log10(self.eps/(3.7*self.ID))**2)

    def pressure_design_thick(self, P_int, P_ext=Q_('0 psig')):
        """Calculate pressure design thickness for given pressure and pipe material.

        Based on B31.3 304.1.

        Parameters
        ----------
        P_int : ureg.Quantity {length: -1, mass: 1, time: -2}
            Internal pressure, absolute
        P_ext : ureg.Quantity {length: -1, mass: 1, time: -2}
            External pressure, absolute

        Returns
        -------
        ureg.Quantity {length: 1}
            Minimum required wall thickness.
        """
        # Check whether S, E, W, and Y are defined
        self._check_SEWY()
        D = self.OD
        # d = self.ID
        S, E, W, Y = self.S, self.E, self.W, self.Y
        P = P_int - P_ext  # Differential pressure across the wall
        t = P * D / (2*(S*E*W + P*Y))
        # TODO add 3b equation handling:
        # t = P * (d+2*c) / (2*(S*E*W-P*(1-Y)))
        if (t >= D/6) or (P/(S*E)) > 0.385:
            logger.error('Calculate design thickness in accordance \
            with B31.3 304.1.2 (b)')
            return None
        tm = t + self.c
        return tm

    def pressure_rating(self):
        """Calculate internal pressure rating.

        Based on B31.3 304.1.

        Returns
        -------
        ureg.Quantity {length: 1}
            Minimum required wall thickness.
        """
        # Check whether S, E, W, and Y are defined
        self._check_SEWY()
        D = self.OD
        S, E, W, Y = self.S, self.E, self.W, self.Y
        t = self.wall - self.c
        P = 2 * t * S * E * W / (D-2*Y*t)
        return P

    def add_SEWY(self, *, S=None, E=None, W=None, Y=None):
        """ Add properties for pressure design calculations.

        Parameters
        ----------
        S : Stress value for pipe material
        E : Quality factor from Table A-1A or A-1B
        W : Weld joint strength reduction factor in accordance
            with 302.3.5 (e)
        Y : coefficient from Table 304.1.1

        Returns
        -------
        None
        """
        self.S, self.E, self.W, self.Y = S, E, W, Y

    def _check_SEWY(self):
        """Check whether following properties S, E, W, Y for stress
        calculations are defined.

        Parameters
        ----------
         S: Stress value for pipe material
         E: Quality factor from Table A-1A or A-1B
         W: Weld joint strength reduction factor in accordance with 302.3.5 (e)
         Y: coefficient from Table 304.1.1

        Returns
        -------
        None
        """
        if self.S is None:
            try:
                self.S = self.material.S
            except AttributeError:
                raise AttributeError(
                    'Stress value S from B31.3 Table A-1  needs to be defined')
        elif self.E is None:
            try:
                self.E
            except AttributeError:
                raise AttributeError('Quality factor E from B31.3 Table '
                                     'A-1A/A-1B needs to be defined')
        elif self.W is None:
            try:
                self.W
            except AttributeError:
                raise AttributeError(
                    'Weld joint stress reduction factor W from B31.3 '
                    '302.3.5(e) needs to be defined')
        elif self.Y is None:
            try:
                self.Y
            except AttributeError:
                raise AttributeError('Coefficient Y from B31.3 Table '
                                     '304.1.1 needs to be defined')
        else:
            if self.S != self.material.S:
                raise AttributeError('Stress value S is different than stress'
                                     'value for material. Check the inputs.')

    def branch_reinforcement(self, BranchPipe, P, beta=Q_('90 deg'), d_1=None,
                             T_r=Q_('0 in')):
        """Calculate branch reinforcement status for given BranchPipe and
        reinforcing ring thickness, Tr.

        Parameters
        ----------
        BranchPipe : Pipe
            Branch pipe/tube instance with S, E, W, Y properties defined
        P : ureg.Quantity {length: -1, mass: 1, time: -2}
            Design pressure
        beta : ureg.Quantity {dimensionless}
            Smaller angle between axes of branch and run
        d_1 : ureg.Quantity {length: 1}
            Effective length removed from pipe at branch (opening for branch)
        T_r : ureg.Quantity {length: 1}
            Minimum thickness of reinforcing ring

        Returns
        -------
        None

        """
        # TODO Rename to reinforcement area?
        t_h = self.pressure_design_thick(P)
        if d_1 is None:
            d_1 = BranchPipe.OD
        D_h = self.OD
        T_h = self.wall
        T_b = BranchPipe.wall
        t_b = BranchPipe.pressure_design_thick(P)
        c = max(self.c, BranchPipe.c)  # Max allowance is used for calculation
        # B31.3 has no specification which allowance to use
        d_2 = half_width(d_1, T_b, T_h, c, D_h)
        A_1 = t_h * d_1 * (2-sin(beta))
        A_2 = (2*d_2-d_1) * (T_h - t_h - c)
        # height of reinforcement zone outside of run pipe
        L_4 = min(2.5*(T_h-c), 2.5*(T_b-c))
        A_3 = 2 * L_4 * (T_b-t_b-c)/sin(beta) * BranchPipe.S / self.S
        print(f'Required Reinforcement Area A_1: {A_1.to(ureg.inch**2):.3g~}')
        A_avail = A_2 + A_3  # Ignoring welding reinforcement
        print(f'Available Area A_2+A_3: {A_avail.to(ureg.inch**2):.3g~}')
        print(f'Weld branch connection is safe: {A_avail>A_1}')

    def __str__(self):
        return f'{self.type}, {self.OD:.3g~}x{self.wall:.3g~}, ' + \
            f'L={self.L:.3g~}'

    # def __str__(self):
    #     return f'{self.D:.3g}" {self.type}'


class Pipe(Tube):
    """NPS pipe class.

    Pipe objects represent actual parts of the pipeline. The Pipe object
    will contain information such as OD, ID, wall thickness, and can be
    used to calculate flow coefficient K that is used for flow calculations.
    """
    def __init__(self, D_nom, SCH=40, L=0*ureg.m, c=Q_('0 mm'),
                 eps=0.0018*ureg.inch):
        """Generate `Pipe` object.

        Parameters
        ----------
        D_nom : float or ureg.Quantity {length: 1}
            Nominal diameter of piping; can be dimensionless or having a unit
            of length.
        SCH : int
            Pipe schedule. Default value is SCH 40 (STD).
        L : ureg.Quantity {length: 1}
            Pipe length
        c : ureg.Quantity {length: 1}
            Sum of the mechanical allowances plus corrosion and erosion
            allowances.
        eps : ureg.Quantity {length: 1}
            Absolute roughness for the tube. Default value for smooth
            pipe.
        """
        if isinstance(D_nom, ureg.Quantity):
            D = D_nom.magnitude
        else:
            D = D_nom
        OD = NPS_table[D]['OD']
        self.SCH = SCH
        # TODO Should I be using get here?
        wall = NPS_table[D].get(SCH)
        super().__init__(OD, wall, L, c, eps)
        self.D = D
        self.type = 'NPS pipe'

    def __str__(self):
        return f'{self.type} {self.D}" SCH {self.SCH}, L={self.L:.3~g}'


class CopperTube(Tube):
    """Copper tube.

    Parameters
    ----------
    OD : ureg.Quantity {length: 1}
        Outer diameter of the tube.
    wall : ureg.Quantity {length: 1}
        Wall thickness of the tube.
    L : ureg.Quantity {length: 1}
        Length of the tube.
    eps : ureg.Quantity {length: 1}
        Absolute roughness for the tube. Default value for smooth
        pipe.
    """
    def __init__(self, D_nom, type_='K', L=0*ureg.m,
                 eps=0.0018*ureg.inch):
        if isinstance(D_nom, ureg.Quantity):
            D = D_nom.magnitude
        else:
            D = D_nom
        OD = COPPER_TABLE[D]['OD']
        wall = COPPER_TABLE[D][type_]
        c = 0 * ureg.inch  # Not affected by corrosion
        super().__init__(OD, wall, L, c, eps)
        self.D = D
        self.type = 'Copper tube Type ' + type_


class VJPipe(Pipe):
    """Vacuum jacketed pipe.
    """
    def __init__(self, D_nom, *, SCH=5, L=0*ureg.m,
                 VJ_D, VJ_SCH=5, c=0*ureg.inch):
        """Generate Vacuum jacketed pipe object.

        Parameters
        ----------
        D_nom : float or ureg.Quantity {length: 1}
            Nominal diameter of the inner pipe.
        SCH : int
            Inner pipe schedule. Default value is SCH 40 (STD).
        L : ureg.Quantity {length: 1}
            Length of the inner pipe.
        VJ_D : float or ureg.Quantity {length: 1}
            Nominal diameter of the vacuum jacket.
        VJ_SCH : int
            Vacuum jacket pipe schedule. Default value is SCH 40 (STD).
        c : ureg.Quantity {length: 1}
            Sum of the mechanical allowances plus corrosion and erosion
            allowances of the inner pipe.
        """
        super().__init__(D_nom, SCH, L)
        self.VJ = Pipe(VJ_D, VJ_SCH, L)
        self.type = 'VJ pipe'

    def info(self):
        return f'NPS {self.D}" SCH {self.SCH} with VJ {self.VJ.D}", ' + \
            f'SCH {self.VJ.SCH}, L={self.L:.3~g}'


class CorrugatedPipe(Tube):
    """Corrugated pipe class.
    """
    def __init__(self, D_nom, L=0*ureg.m):
        """Generate corrugated pipe object.

        Parameters
        ----------
        D_nom : ureg.Quantity {length: 1}
            Nominal diameter of the inner pipe.
        L : ureg.Quantity {length: 1}
            Length of the pipe.
        """
        # TODO DRY
        OD = D_nom
        logger.debug('For corrugated piping assumed OD = D')
        wall = 0 * ureg.m
        c = 0 * ureg.inch
        super().__init__(OD, wall, L, c)
        self.type = 'Corrugated pipe'
        logger.debug('For corrugated piping assumed wall = 0')

    def K(self, Re_):
        return 4 * super().K(Re_)  # Multiplier 4 is used for corrugated pipe

    def branch_reinforcement(self):
        raise NotImplementedError('Branch reinforcement not implemented for'
                                  ' corrugated pipe')

    def pressure_design_thick(self):
        raise NotImplementedError('Pressure design thickness not implemented'
                                  ' for corrugated pipe')

    def __str__(self):
        return f'Corrugated pipe D={self.OD:.3g~}, L={self.L:.3g~}'


class Entrance ():
    """Pipe entrance, flush, sharp edged.
    """
    def __init__(self, ID):
        """Generate an pipe entrance.

        Parameters
        ----------
        ID : ureg.Quantity {length: 1}
            Inside diameter of the entrance.
        """
        self.ID = ID
        self.area = Tube._calculate_area(self)
        self.L = 0*ureg.m
        self.type = 'Entrance'
        self.volume = 0 * ureg.ft**3

    def K(self):
        return 0.5  # Crane TP-410, A-29

    def __str__(self):
        return f'{self.type}, {self.ID:.3g~}'

    # def __str__(self):
    #     return self.type


class Exit (Entrance):
    """Pipe exit, projecting or sharp-edged, or rounded.
    """
    def __init__(self, ID):
        """Generate pipe exit.

        Parameters
        ----------
        ID : ureg.Quantity {length: 1}
            Inside diameter of the exit.
        """
        self.ID = ID
        self.area = Tube._calculate_area(self)
        self.L = 0*ureg.m
        self.type = 'Exit'
        self.volume = 0 * ureg.ft**3

    def K(self):
        return 1  # Crane TP-410, A-29

    def __str__(self):
        return f'Exit opening, {self.ID:.3g~}'


class Orifice():
    """Square-edged orifice plate
    """
    def __init__(self, ID):
        """Generate orifice.

        Parameters
        ----------
        ID : ureg.Quantity {length: 1}
            Inside diameter of the orifice.
        """
        self.Cd = 0.61  # Thin sharp edged orifice plate
        self.ID = ID
        self.area = Pipe._calculate_area(self)
        self.type = 'Orifice'
        self.volume = 0 * ureg.ft**3

    def K(self):
        return 1/self.Cd**2

    def __str__(self):
        return f'Orifice, {self.ID:.3g~}'

    # def __str__(self):
    #     return f'{self.ID:.3g~} orifice'


class ConicOrifice(Orifice):
    """Conic orifice
    """
    def __init__(self, D, ID):
        """Generate conic orifice.

        Parameters
        ----------
        D :
        ID : ureg.Quantity {length: 1}
            Inside diameter of the orifice.
        """
        super().__init__(ID)
        if NPS_table[D]['OD'] >= 1*ureg.inch:
            # For a smaller diameter using value for
            # square-edged plate (unfounded assumption)
            self.Cd = 0.73
            # Flow Measurements Engineering Handbook, Table 9.1, p. 9.16
        self.type = 'Conic orifice'

    def __str__(self):
        return f'Conic orifice, {self.ID:.3g~}'

    # def __str__(self):
    #     return f'{self.ID:.3g~} conic orifice'


class Annulus():
    """"Annulus or tube in tube.

    Parameters
    ----------
    D1 : ureg.Quantity {length: 1}
        ID of the outer tube.
    D2 : ureg.Quantity {length: 1}
        OD of the inner tube.
    L : ureg.Quantity {length: 1}
        Length of the annulus tube.
    """
    def __init__(self, D1, D2, L=Q_('0m'), eps=0.0018*ureg.inch):
        self.D1 = D1
        self.D2 = D2
        assert D1 > D2, 'D1 should be larger than D2'
        self.L = L
        assert D1 > D2
        self.area = pi / 4 * (D1**2 - D2**2)
        self.volume = Tube._calculate_volume(self)
        self.ID = D1 - D2  # Hydraulic diameter
        self.eps = eps

    def K(self, Re_):
        return Tube.K(self, Re_)

    def __str__(self):
        return f'Annulus D1={self.D1:.3g~}, D2={self.D2:.3g~}, L={self.L:.3g~}'

    # def __str__(self):
    #     return f'{self.D1:.3g~} x {self.D2:.3g~} annulus'


class Elbow(Tube):
    """
    NPS Tee fitting.
    MRO makes method K from PipeElbow class to override method from Pipe class.
    """
    def __init__(self, OD, wall=0*ureg.inch, c=0*ureg.inch, R_D=1.5, N=1,
                 angle=90*ureg.deg):
        """Generate a tube elbow object.

        Parameters
        ----------
        OD : ureg.Quantity {length: 1}
            Outer diameter of the tube.
        wall : ureg.Quantity {length: 1}
            Wall thickness of the tube.
        R_D : ureg.Quantity {length: 1}
            Elbow radius/diameter ratio
        N : int
            Number of elbows in the pipeline
        angle : ureg.Quantity {dimensionless}
            Number of elbows in the pipeline
        """
        self.R_D = R_D
        self.N = N
        self.angle = angle
        super().__init__(OD, wall, L=0*ureg.m, c=c)
        self.L = R_D*self.ID*angle
        self.volume = self._calculate_volume()
        self.type = 'Tube elbow'

    def K(self, Re_):
        """
        Pressure drop in an elbow fitting.
        Based on Handbook of Hydraulic Resistance by I.E. Idelchik.
        """
        if self.angle <= 70*ureg.deg:
            A1 = 0.9*sin(self.angle)
        elif self.angle == 90*ureg.deg:
            A1 = 1
        elif self.angle >= 100*ureg.deg:
            A1 = 0.7+0.35*self.angle/(90*ureg.deg)
        else:
            logger.error('''Improper bend angle for elbow.
            90 degrees used instead: {}'''.format(self.angle))
            A1 = 1

        if self.R_D < 1:
            B1 = 0.21*(self.R_D)**(-0.25)
        else:
            B1 = 0.21*(self.R_D)**(-0.5)

        C1 = 1  # use different value for non-axis symmetric
        # Friction losses in the elbow
        K_frict = super().K(Re_)
        return (A1*B1*C1+K_frict)*self.N

    def __str__(self):
        return f'{self.N}x {self.type}, {self.OD}"x{self.wall}", ' + \
            f'{self.angle.to(ureg.deg)}, R_D = {self.R_D}'


class PipeElbow(Elbow, Pipe):
    """
    NPS Elbow fitting.
    MRO makes method K from Elbow class to override method from Pipe class.
    """
    def __init__(self, D_nom, SCH=40, c=0*ureg.inch, R_D=1.5, N=1, angle=90*ureg.deg):
        """Generate a pipe elbow object.

        Parameters
        ----------
        D_nom : float or ureg.Quantity {length: 1}
            Nominal diameter of piping; can be dimensionless or having a unit
            of length.
        SCH : int
            Pipe schedule. Default value is SCH 40 (STD).
        R_D : ureg.Quantity {length: 1}
            Elbow radius/diameter ratio
        N : int
            Number of elbows in the pipeline
        angle : ureg.Quantity {dimensionless}
            Number of elbows in the pipeline
        """
        # D_nom and SCH go as positional arguments to Pipe __init__
        super().__init__(D_nom, SCH, c=c, R_D=R_D, N=N, angle=angle)
        self.type = 'NPS elbow'

    def __str__(self):
        return f'{self.N}x {self.type}, {self.D}" SCH {self.SCH}, ' + \
            f'{self.angle.to(ureg.deg)}, R_D = {self.R_D}'


class Tee(Tube):
    """
    Tee fitting.
    """
    def __init__(self, OD, wall=0*ureg.inch, c=0*ureg.inch, direction='thru',
                 N=1):
        """Generate a tee.

        Parameters
        ----------
        OD : ureg.Quantity {length: 1}
            Outer diameter of the tube.
        wall : ureg.Quantity {length: 1}
            Wall thickness of the tube.
        c : ureg.Quantity {length: 1}
            Sum of the mechanical allowances plus corrosion and erosion
            allowances of the inner pipe.
        N : int
            Number of tees in the pipeline
        """
        if direction in ['thru', 'through', 'run']:
            self.direction = 'run'
        elif direction in ['branch', 'side']:
            self.direction = 'branch'
        else:
            logger.error('''Tee direction is not recognized,
                         try "thru" or "branch": {}'''.format(direction))
        L = 0*ureg.m
        super().__init__(OD, wall, L, c)
        self.N = N
        self.type = 'Tube tee'

    def K(self):
        if self.direction == 'run':
            K_ = 20*self.f_T()  # Crane TP-410 p. A-29
        elif self.direction == 'branch':
            K_ = 60*self.f_T()  # Crane TP-410 p. A-29
        return self.N * K_

    def __str__(self):
        return f'{self.type}, {self.OD}x{self.wall}, {self.direction}'


class PipeTee(Tee, Pipe):
    """
    NPS Tee fitting.
    MRO makes method K from Tee class to override method from Pipe class.
    """
    def __init__(self, D_nom, SCH=40, c=0*ureg.inch, direction='thru', N=1):
        # D_nom and SCH go as positional arguments to Pipe __init__
        super().__init__(D_nom, SCH, c, direction)
        self.type = 'NPS tee'

    def info(self):
        return f'{self.type}, {self.D}" SCH {self.SCH}, ' + \
            f'{self.direction}'


class Valve():
    """
    Generic valve with known Cv.
    """
    def __init__(self, D, Cv):
        # TODO DRY
        self.D = D
        self._Cv = Cv
        self.OD = None
        self.ID = self.D
        self.area = Pipe._calculate_area(self)
        self.L = None
        self.type = 'Valve'
        self.volume = 0 * ureg.ft**3

    def K(self):
        return Cv_to_K(self._Cv, self.D)

    def __str__(self):
        return f'{self.type}, {self.D}", Cv = {self._Cv:.3g}'


# class GlobeValve(Pipe):
#     """
#     Globe valve.
#     """
#     def __init__(self, D):
#         super().__init__(D, None, None)
#         # ID for the valve is assumed equal to SCH40 ID:
#         self.ID = self.OD - 2*NPS_table[D].get(40)
#         self.type = 'Globe valve'
#         self._K = 340*self.f_T()  # Horizontal ball valve with beta = 1

#     @property
#     def volume(self):
#         return 0 * ureg.ft**3

#     def __str__(self):
#         return f'{self.type}, {self.D}"'


# class VCone(Pipe):
#     """
#     McCrometer V-Cone flowmeter.
#     """
#     def __init__(self, D, beta, Cf, SCH=40):
#         super().__init__(D, SCH, None)
#         self.beta = beta
#         self._Cf = Cf
#         self.type = 'V-cone flow meter'
#         # Equation is reverse-engineered from McCrometer V-Cone equations
#         self._K = 1/(self.beta**2/(1-self.beta**4)**0.5*self._Cf)**2


class Contraction():
    """
    Sudden and gradual contraction based on Crane TP-410.
    """
    def __init__(self, ID1, ID2, theta=ureg('180 deg')):
        """
        ID1 : upstream pipe ID
        ID2 : downstream pipe ID
        theta : contraction angle
        """
        self.beta = beta(ID1, ID2)
        self.theta = theta
        self.type = 'Contraction'
        self.L = None
        self.OD = None
        self.ID1 = ID1
        self.ID2 = ID2
        self.ID = min(ID1, ID2)
        self.area = Pipe._calculate_area(self)
        self.L = abs(ID1 - ID2) / tan(theta/2)
        self.volume = pi * self.L / 3 * (ID1**2 + ID1*ID2 + ID2**2)

    def K(self):
        if self.theta <= 45*ureg.deg:
            K_ = 0.8 * sin(self.theta/2) * (1-self.beta**2)
            # Crane TP-410 A-26, Formula 1 for K1 (smaller dia)
        elif self.theta <= 180*ureg.deg:
            K_ = 0.5 * (1-self.beta**2) * sqrt(sin(self.theta/2))
            # Crane TP-410 A-26, Formula 2 for K1 (smaller dia)
        else:
            logger.error(f'Theta cannot be greater than {180*ureg.deg} '
                         f'(sudden contraction): {self.theta}')
        return K_

    def __str__(self):
        return f'{self.type}, {self.theta.to(ureg.deg)} from {self.ID1} ' + \
            f'to {self.ID2}'

    # def __str__(self):
    #     return f'{self.type}'


class Enlargement(Contraction):
    """
    Sudden and gradual enlargement based on Crane TP-410.
    """
    def __init__(self, ID1, ID2, theta=ureg('180 deg')):
        """
        ID1 : upstream pipe ID
        ID2 : downstream pipe ID
        theta : contraction angle
        """
        super().__init__(ID1, ID2, theta)
        self.type = 'Enlargement'

    def K(self):
        if self.theta <= 45*ureg.deg:
            K_ = 2.6 * sin(self.theta/2) * (1-self.beta**2)**2
            # Crane TP-410 A-26, Formula 3 for K1 (smaller dia)
        elif self.theta <= 180*ureg.deg:
            K_ = (1-self.beta**2)**2
            # Crane TP-410 A-26, Formula 4 for K1 (smaller dia)
        else:
            logger.error(f'Theta cannot be greater than {180*ureg.deg} \
            (sudden enlargement): {self.theta}')
        return K_


class Piping(MutableSequence):
    '''
    Piping system defined by initial conditions and structure of
    pipe elements.
    Piping is modeled as a list of Pipe objects with given conditions at the
    beginning. Implemented methods allow to calculate pressure drop for given
    mass flow rate or mass flow rate for given pressure drop using lumped
    Darcy equation. All flow coefficients K are converted to the same base and
    added together to calculate single K value for the whole piping. This K
    value is used with Darcy equation to calculate pressure drop or mass flow.
    '''
    pipe_type = (Pipe, VJPipe, CorrugatedPipe, Tube, Annulus, Elbow)
    def __init__(self, *elements):
        self._elements = list(elements)

    def volume(self):
        result = []
        # TODO remove elbows and tees after merge
        for pipe in self._elements:
            if isinstance(pipe, Piping.pipe_type):
                result.append((str(pipe),
                               f'{pipe.L.to(ureg.ft).magnitude:.3g}',
                               f'{pipe.volume.to(ureg.ft**3).magnitude:.3g}'))
        return result

    def stored_energy(self, fluid):
        """Calculate stored energy of the piping.

        Uses 8 diameters rule as per ASME PCC-2 2018 501-IV-3 (a)."""
        largest_tube = max(self, key=lambda tube: tube.ID)
        volume = pi * largest_tube.ID**2 / 4 * 8 * largest_tube.L
        return stored_energy(fluid, volume)

    def __str__(self):
        return '\n'.join([el.__str__() for el in self._elements])

    def __delitem__(self, idx):
        del self._elements[idx]

    def __getitem__(self, idx):
        return self._elements[idx]

    def __len__(self):
        return len(self._elements)

    def __setitem__(self, idx, value):
        self._elements[idx] = value

    def insert(self, idx, value):
        self._elements.insert(idx, value)


class ParallelPlateRelief:
    def __init__(self, Springs, Plate, Supply_pipe):
        """
        Initiate Parallel Plate instance.
        Parallel Plate relief valve designed by Fermilab

        Springs: dictionary containing 'N' - number of springs, 'k' - spring rate, and 'dx_precomp' - pre-compression length
        Plate: dictionary containing 'OD_plate' - OD of the plate, 'OD_O_ring' - OD of the O-Ring installed, and 'W_plate' - plate weight
        Supply_pipe: Pipe/Tube object of upstream pipe
        """
        self.Springs = Springs
        self.Plate = Plate
        self.Supply_pipe = Supply_pipe

    def P_set(self):
        """"
        Calculate set (lift) pressure of the Parallel Plate relief
        """
        # Before lifting pressure affects area within O-Ring OD
        A_lift = pi * self.Plate['OD_O_ring']**2 / 4
        F_precomp = self.Springs['N'] * self.Springs['k'] *\
                    self.Springs['dx_precomp']
        # Force required to lift the plate
        F_lift = F_precomp + self.Plate['W_plate']
        self.F_lift = F_lift
        P_set = F_lift / A_lift
        return P_set.to(ureg.psi)

    def P_open(self):
        """
        Calculate pressure required to fully open Parallel Plate relief
        """
        # compression required to provide vent area equal to supply pipe area
        dx_open = self.Supply_pipe.area / (pi*self.Plate['OD_plate'])
        # Force at fully open
        F_open = self.F_lift = self.Springs['N'] * self.Springs['k'] * dx_open
        # At fully open pressure is distributed as:
        # Full pressure for up to supply pipe diameter
        # Linear fall off up to plate OD
        A_open = self.Supply_pipe.area + pi/8*(self.Plate['OD_plate']**2 -
                                               self.Supply_pipe.ID**2)
        P_open = F_open / A_open
        return P_open.to(ureg.psi)


# Supporting functions used for flow rate and pressure drop calculations.
def f_Darcy(Re_, eps_r, method='churchill'):
    """Calculate Darcy friction factor using Serghide solution to
    Colebrook equation.

    See Crane TP-410 2013, equation 6-6.

    Parameters
    ----------
    Re_ : float or Quantity {dimensionless}
        Reynolds number
    eps_r : float or Quantity {dimensionless}
        Absolute roughness of the pipe
    method : str
        Friction factor formula name

    Returns
    -------
    float
        Darcy friction coefficient
    """
    if method == 'churchill':
        return churchill(Re_, eps_r)
    elif method == 'serghide':
        return serghide(Re_, eps_r)

def churchill(Re_, eps_r):
    """Calculate Darcy friction factor using modified Churchill formula.
    See 8.5.2 of "Pipe flow, A Practical and Comprehensive Guide", Rennels,
    Hobart, Hudson, 2012.

    Parameters
    ----------
    Re_ : float or Quantity {dimensionless}
        Reynolds number
    eps_r : float or Quantity {dimensionless}
        Relative roughness of the pipe

    Returns
    -------
    float
        Darcy friction coefficient
    """
    A1 = 0.883 * log(Re_)**1.282 / Re_**1.007
    A2 = 0.27 * eps_r
    A3 = - 110 * eps_r / Re_
    A = (0.8687 * log(A1+A2+A3))**16
    B = (13269 / Re_)**16
    f = ((64/Re_)**12 + 1/(A+B)**(3/2))**(1/12)
    return f

def serghide(Re_, eps_r):
    """Calculate Darcy friction factor using Serghide solution to
    Colebrook equation.

    See Crane TP-410 2013, equation 6-6.

    Parameters
    ----------
    Re_ : float or Quantity {dimensionless}
        Reynolds number
    eps_r : float or Quantity {dimensionless}
        Relative roughness of the pipe

    Returns
    -------
    float
        Darcy friction coefficient
    """
    A = -2 * log10(eps_r/3.7+12/Re_)
    B = -2 * log10(eps_r/3.7+2.51*A/Re_)
    C = -2 * log10(eps_r/3.7+2.51*B/Re_)
    f = (A - (B-A)**2/(C-2*B+A))**(-2)
    return f

def K_piping(m_dot, fluid, piping):
    """Calculate resistance coefficient converted to the area of the first element.

    Returns
    -------
    tuple
        K0 : converted resistance coefficient of the piping
        A0 : area of the first element, basis for conversion
    """
    K0 = 0*ureg.dimensionless
    try:
        A0 = piping[0].area  # using area of the first element as base
    except IndexError:
        raise IndexError('Piping has no elements! '
                            'Use Piping.add to add sections to piping.')
    for element in piping:
        Re_ = Re(fluid, m_dot, element.ID, element.area)
        if isinstance(element, Piping.pipe_type) and \
            not isinstance(element, Tee):
            K_el = element.K(Re_) * (A0/element.area)**2
            K0 += K_el
        else:
            K_el = element.K() * (A0/element.area)**2
            K0 += K_el
    return (K0.to_base_units(), A0)

def rc(fluid):
    """Calculate critical pressure drop for the given fluid."""
    k = fluid.gamma
    rc = (2/(k+1))**(k/(k-1))
    return rc

def dP_Darcy(K, rho, w):
    '''
    Darcy equation for pressure drop.
    K - resistance coefficient
    rho - density of flow at entrance
    w - flow speed
    '''
    d_P = K*rho*w**2/2
    return d_P.to(ureg.psi)

def dP_incomp(m_dot, fluid, piping):
    """Calculate pressure drop of incompressible flow through piping.
    Lumped method using Darcy equation is used.
    """
    P_0 = fluid.P
    T_0 = fluid.T
    rho_0 = fluid.Dmass
    K, area = K_piping(m_dot, fluid, piping)
    w = m_dot / (rho_0*area)
    dP = dP_Darcy(K, rho_0, w)  # first iteration
    P_out = P_0 - dP
    if fluid.Q < 0 or dP/P_0 <= 0.1:  # if q<0 then fluid is a liquid
        return dP
    elif dP/P_0 <= 0.4:
        TempState = fluid.copy()
        TempState.update('T', T_0, 'P', P_out)
        rho_out = TempState.Dmass
        rho_ave = (rho_0+rho_out) / 2
        # Comprehensive average properties, Rennels 4.2.2.2
        K_corr = K - rho_ave**2/rho_0**2 + rho_ave**2/rho_out**2
        w = m_dot/(rho_ave*area)
        return dP_Darcy(K_corr, rho_ave, w)
    else:
        raise HydraulicError(f'Estimated pressure drop {dP/P_0:.3g~} exceeds 40 % limit for'
                             ' incompressible flow.')


def m_dot_incomp(fluid, piping, P_out=0*ureg.psig, guess=1*ureg.g/ureg.s):
    '''Calculate mass flow through the piping using initial conditions
    at the beginning of piping.

    Calculation is based on Crane TP-410, p. 1.9.
    Net expansion factor Y is conservatively assumed as 1.
    Mass flow is calculated using Darcy equation.

    Parameters
    ----------
    P_out : ureg.Quantity {length: -1, mass: 1, time: -1}
        Exit pressure of the piping.
    guess : ureg.Quantity {mass: 1, time: -1}
        guess value for the mass flow rate

    Returns
    -------
    ureg.Quantity : {mass: 1, time: -1}
    '''
    P_0 = fluid.P
    if P_0 <= P_out:
        raise HydraulicError(f'Input pressure less or equal to output: \
        {P_0.to(ureg.Pa):.3g}, {P_out.to(ureg.Pa):.3g}')
    def to_solve(m_dot_gs, P_in_Pa, P_out_Pa):
        dP_calc = dP_incomp(m_dot_gs*ureg.g/ureg.s, fluid, piping)
        dP_given = P_in_Pa - P_out_Pa
        return dP_given - dP_calc.m_as(ureg.Pa)
    P_in_Pa = P_0.m_as(ureg.Pa)
    P_out_Pa = P_out.m_as(ureg.Pa)
    args = (P_in_Pa, P_out_Pa)  # arguments passed to to_solve
    x0 = guess.m_as(ureg.g/ureg.s)
    x1 = x0 * 2  # Usually doubling the flow is OK estimate
    solution = root_scalar(to_solve, args, x0=x0, x1=x1)
    m_dot_ = solution.root * ureg.g/ureg.s
    return m_dot_

# def _solver_func(self, P_in_Pa, m_dot, P_out_act):
#     """
#     Solver function for calculating upstream pressure given flow and
#     downstream pressure.

#     :P_in_Pa: input pressure in Pa, float
#     :args: calculation parameters:
#         :m_dot: mass flow
#         :P_out_act: actual downstream pressure
#     """
#     P_in = Q_(P_in_Pa, ureg.Pa)
#     self.fluid.update('P', P_in, 'Smass', self.fluid.Smass)
#     P_out_calc = P_in - self.dP(m_dot)
#     P_out_calc_Pa = P_out_calc.to(ureg.Pa).magnitude
#     P_out_act_Pa = P_out_act.to(ureg.Pa).magnitude
#     return P_out_calc_Pa - P_out_act_Pa

# def P_in(self, m_dot, P_out=ureg('0 psig'), P_min=ureg('0 psig'),
#             P_max=ureg('200 bar')):
#     """
#     Calculate upstream pressure given mass flow and downstream pressure.

#     :m_dot: mass flow
#     :P_out: downstream pressure
#     :P_min: min expected pressure; should be lower than actual value
#     :P_max: max expected pressure; should be higher than actual value
#     """
#     args = (m_dot, P_out)  # arguments passed to _solver_func
#     # Convert pressure to dimensionless form
#     P_min_Pa = P_min.to(ureg.Pa).magnitude
#     P_max_Pa = P_max.to(ureg.Pa).magnitude
#     bracket = [P_min_Pa, P_max_Pa]
#     logger_level = logger.getEffectiveLevel()
#     # ERROR and CRITICAL only will be shown; WARNING is suppressed
#     logger.setLevel(40)
#     solution = root_scalar(self._solver_func, args, bracket=bracket,
#                             method='brentq')
#     logger.setLevel(logger_level)
#     P_in = Q_(solution.root, ureg.Pa)
#     self.fluid.update('P', P_in, 'Smass', self.fluid.Smass)
#     logger.debug(f'Comparing pressure drop:\n    dP method:\n        \
#     {self.dP(m_dot).to(ureg.psi)}\n    \
#     P_in - P_out:\n        \
#     {(P_in-P_out).to(ureg.psi)}')
#     logger.info(f'Calculated initial pressure: {P_in.to(ureg.psi):.1~f}')

def m_dot_isot(fluid, pipe, P_out=0*ureg.psig, m_dot_g=1*ureg.g/ureg.s, tol=1e-6):
    """Calculate mass flow rate through piping for isothermal compressible
    flow.

    See 4.4 of "Pipe flow, A Practical and Comprehensive Guide", Rennels,
    Hobart, Hudson, 2012.

    Parameters
    ----------
    fluid : ThermState
        Inlet fluid conditions
    pipe : Pipe
    P_out : ureg.Quantity {length: -1, mass: 1, time: -1}
        Exit pressure of the piping.

    Returns
    -------
    Quantity {length: -1, mass: 1, time: -2}
        Pressure drop
    """
    P1 = fluid.P
    P2 = P_out
    T = fluid.T
    R = fluid.specific_gas_constant
    K = pipe.K(Re(fluid, m_dot_g, pipe.ID, pipe.area))
    A = pipe.area
    m_dot = A * ((P1**2-P2**2)/(R*T*(2*log(P1/P2)+K)))**0.5
    if abs(m_dot-m_dot_g)/m_dot > tol:
        m_dot = m_dot_isot(fluid, pipe, P_out, m_dot, tol)
    return m_dot.to(ureg.g/ureg.s)

def dP_isot(m_dot, fluid, pipe):
    """Calculate pressure drop through piping for isothermal compressible
    flow.

    See 4.4 of "Pipe flow, A Practical and Comprehensive Guide", Rennels,
    Hobart, Hudson, 2012.

    Parameters
    ----------
    m_dot : Quantity {mass: 1, time: -1}
        mass flow rate
    fluid : ThermState
        Inlet fluid conditions
    pipe : Pipe

    Returns
    -------
    Quantity {length: -1, mass: 1, time: -2}
        Pressure drop
    """
    R = fluid.specific_gas_constant
    T = fluid.T
    P1 = fluid.P
    A = pipe.area
    K = pipe.K(Re(fluid, m_dot, pipe.ID, pipe.area))
    def to_solve_sq(P2sq_):
        P1_ = P1.m_as(ureg.Pa)
        m_dot_ = m_dot.m_as(ureg.kg/ureg.s)
        R_ = R.m_as(ureg.J/ureg.kg/ureg.K)
        T_ = T.m_as(ureg.K)
        A_ = A.m_as(ureg.m**2)
        K_ = float(K)
        B = m_dot_**2 * R_ * T_ / A_**2
        sq_diff = B * (2*log(P1_/(P2sq_)**0.5)+K_)
        return P2sq_ - P1_**2 + sq_diff, 1 - B/P2sq_, B/P2sq_**2
    x0 = P1.m_as(ureg.Pa)**2
    x1 = 0.9 * x0
    bracket = [1e-9, x0]
    methods = ['secant', 'newton', 'halley']
    for method in methods:
        try:
            solution = root_scalar(to_solve_sq, x0=x0, x1=x1, fprime=True,
                                    fprime2=True, bracket=bracket, method=method)
            P_2_root_sq = solution.root**0.5 * ureg.Pa
            logger.debug(method, solution.iterations)
            break
        except TypeError:
            logger.debug(f'{method} method failed for square solve')
            P_2_root_sq = None
    if P_2_root_sq is None:
        raise HydraulicError('No solution for isothermal pressure drop found')
    else:
        return P1 - P_2_root_sq


def Mach(fluid, v):
    """Calculate Mach number for given static conditions of the fluid.


    Parameters:
    -----------
    fluid : ThermState
        Fluid state at static temperature and pressure
    v : Quantity, {length: 1, time: -1}
    """
    M = v / fluid.speed_sound
    return M.to_base_units()

def Mach_total(fluid, m_dot, area):
    """Calculate Mach number for total temperature and pressure.

    Parameters:
    -----------
    fluid : ThermState
        Fluid state at total temperature and pressure
    """
    k = float(fluid.gamma)
    v = velocity(fluid, m_dot, area)
    M_core = float(Mach(fluid, v))

    def M_sq_total(Msq, M_core, k):
        return M_core**2 * (1+Msq*(k-1)/2)**((k+1)/(k-1))
    def to_solve(Msq):
        return Msq - M_sq_total(Msq, M_core, k)

    x0 = M_core**2
    x1 = 0.9 * x0
    bracket = [0, 1]
    solution = root_scalar(to_solve, x0=x0, x1=x1)
    M_root = solution.root**0.5
    T = fluid.T - v**2 / fluid.Cpmass
    if T < 0*ureg.K:
        raise HydraulicError(f'Negative static temperature: {T:.3g~}. Required \
        speed {v:.3g~} cannot be achieved.')
    if isinstance(M_root, complex):
        raise HydraulicError(f'No real solutions for Mach number found.')
    return M_root

def K_lim(M, k):
    """Calculate max resistance coefficient of a pipe.
    """
    A = (1-M**2) / (k*M**2)
    B = (k+1)/(2*k)
    C = (k+1) * M**2
    D = 2 + (k-1) * M**2
    K = A + B * log(C/D)
    return K

def M_from_K_lim(K, k):
    if K < 0:
        raise HydraulicError(f"Resistance coefficient value can't be less \
        than 0: {K}")
    K_ = float(K)
    def to_solve(M):
        return K_ - K_lim(M, k)
    x0 = 0.9
    x1 = 0.5
    bracket = [1e-15, 1]
    try:
        solution = root_scalar(to_solve, bracket=bracket, method='brentq')
        logger.debug('Brentq failed for M_from_K_lim')
    except ValueError:
        solution = root_scalar(to_solve, x0=x0, x1=x1)
    M = solution.root
    return M

def M_complex(M, k):
    return 1 + M**2 * (k-1) /2

def P_from_M(P1, M1, M2, k):
    """Calculate static pressure from inlet static pressure and Mach numbers."""
    PM1 = P1 * M1 * M_complex(M1, k)**0.5
    P2 = PM1 / (M2*M_complex(M2, k)**0.5)
    return P2

def P_total(P, M, k):
    """Calculate total pressure from static pressure."""
    return P * M_complex(M, k)**(k/(k-1))

def P_crit(P, M, k):
    """Calculate critical(sonic) pressure for the given static pressure and Ma."""
    M_crit_comp = (k+1) / (2+(k-1)*M**2)
    P_c = P * M / M_crit_comp**0.5
    return P_c

def dP_adiab(m_dot, fluid, pipe):
    """Calculate pressure drop for isentropic flow given total inlet conditions.

    """
    M = Mach_total(fluid, m_dot, pipe.area)
    K_limit = K_lim(M, fluid.gamma)
    Re_ = Re(fluid, m_dot, pipe.ID, pipe.area)
    K_pipe = pipe.K(Re_)
    K_left = K_limit - K_pipe
    if K_left < 0:
        raise ChokedFlow(f'Flow is choked at K={float(K_limit):.3g} with given '
                         f'K={float(K_pipe):.3g}. Reduce hydraulic resistance or'
                         ' mass flow.')
    M_end = M_from_K_lim(K_left, fluid.gamma)
    P_static_end = P_from_M(fluid.P, M, M_end, fluid.gamma)
    P_total_end = P_total(P_static_end, M, fluid.gamma)
    return fluid.P - P_total_end

def m_dot_adiab(fluid, pipe, P_out=0*ureg.psig, m_dot_g=1*ureg.g/ureg.s, state='total'):
    """Calculate mass flow rate through piping for adiabatic compressible
    flow.

    Parameters
    ----------
    fluid : ThermState
        Inlet fluid conditions
    pipe : Pipe
    P_out : Quantity {length: -1, mass: 1, time: -2}
        Outlet pressure

    Returns
    -------
    Quantity {mass: 1, time: -1}
        mass flow rate
    """
    k = fluid.gamma
    P1 = fluid.P
    P2 = P_out
    m_dot_g_ = m_dot_g.m_as(ureg.kg/ureg.s)
    def to_solve(m_dot_):
        m_dot = m_dot_ * ureg.kg/ureg.s
        if state == 'total':
            try:
                M1 = Mach_total(fluid, m_dot, pipe.area)
            except HydraulicError:
                return -1
        elif state =='static':
            v = velocity(fluid, m_dot, pipe.area)
            M1 = Mach(fluid, v)
        P_crit1_ = P_crit(P1, M1, k).m_as(ureg.Pa)
        K_lim1 = K_lim(M1, k)
        Re_ = Re(fluid, m_dot_g, pipe.ID, pipe.area)
        K_lim2 = K_lim1 - pipe.K(Re_)
        if K_lim2 < 0:
            return -1
        M2 = M_from_K_lim(K_lim2, k)
        P_crit2_ = P_crit(P2, M2, k).m_as(ureg.Pa)
        result = P_crit1_ - P_crit2_
        return result
    bracket = [1e-10, 1e10]  # Limits search range
    solution = root_scalar(to_solve, bracket=bracket)
    m_dot = solution.root * ureg.kg/ureg.s
    return m_dot

def K_to_Cv(K, ID):
    """
    Calculate flow coefficient Cv based on resistance coefficient value K.
    Based on definition:
    Cv = Q*sqrt(rho/(d_P*rho_w))
    where Q - volumetric flow, rho - flow density
    rho_w - water density at 60 F
    d_P - pressure drop through the valve.
    [Cv] = gal/(min*(psi)**0.5)
    """
    A = pi*ID**2/4
    rho_w = 999*ureg('kg/m**3')  # Water density at 60 F
    # TODO move density to global const (or use ThermState)
    # Based on Crane TP-410 p. 2-10 and Darcy equation:
    Cv = A*(2/(K*rho_w))**0.5
    Cv.ito(ureg('gal/(min*(psi)**0.5)'))  # Convention accepted in the US
    return Cv


def Cv_to_K(Cv, ID):
    """
    Calculate resistance coefficient K based on flow coefficient value Cv.
    Based on definition:
    Cv = Q*sqrt(rho/(d_P*rho_w))
    where Q - volumetric flow, rho - flow density
    rho_w - water density at 60 F
    d_P - pressure drop through the valve.
    [Cv] = gal/(min*(psi)**0.5)
    """
    if isinstance(Cv, (int, float)):  # Check if dimensionless input
        Cv = Cv*ureg('gal/(min*(psi)**0.5)')  # Convention accepted in the US
    A = pi*ID**2/4
    rho_w = 999*ureg('kg/m**3')  # Water density at 60 F
    # Based on Crane TP-410 p. 2-10 and Darcy equation:
    K = 2*A**2/(Cv**2*rho_w)
    return K.to(ureg.dimensionless)


def beta(d1, d2):
    """
    Calculate beta = d/D for contraction or enlargement.
    """
    return min(d1, d2)/max(d1, d2)


def equivalent_orifice(m_dot, dP, fluid=AIR):
    """
    Calculate ID for the equivalent square edge orifice (Cd = 0.61) for given
    flow and pressure drop.
    """
    Cd = 0.61
    rho = fluid.Dmass
    ID = 2 * (m_dot/(pi*Cd*(2*dP*rho)**0.5))**0.5
    return ID.to(ureg.inch)


def half_width(d_1, T_b, T_h, c, D_h):
    """
    Calculate 'half width' of reinforcement zone per B31.3 304.3.3.
    """
    d_2_a = (T_b-c) + (T_h-c) + d_1/2
    return min(max(d_1, d_2_a), D_h)


def velocity(fluid, m_dot, area):
    """Calculate velocity of fluid with given local parameters."""
    return m_dot/(area*fluid.Dmass)
