### Pressure drop and heat transfer calculation

from math import pi, sin, log, log10, sqrt, tan, tanh     
from .std_conditions import ureg, Q_, P_NTP
from .functions import Re, Ra, Nu_vcyl, Nu_hcyl     
from .functions import Material, Property     
from .functions import nist_property, conduction_cyl     
from scipy.optimize import root_scalar, minimize  
from .functions import AIR
from .functions import heat_trans_coef, Ra, Nu_vcyl, Nu_hcyl, Pr
from .cp_wrapper import ThermState
from .piping import Mach, Mach_total, K_lim, ChokedFlow, HydraulicError, velocity, dP_Darcy, dP_adiab, Pipe, Tube, CopperTube
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import CryoToolBox as ht
import pandas as pd

refprop_data = pd.read_csv(r'C:\Users\rbeckwit\Documents\Python Scripts\Heated_pipe\Heated_pipe_part2\refprop_data.txt', sep ='\s+', engine = 'python')

rp_temp = refprop_data['Temperature']
rp_density = refprop_data['Density']
rp_enthalpy = refprop_data['Enthalpy']

# #Tell pdoc not to include in documentation
# __pdoc__ = {
#     'pipe_isolation.__init__': False,  # Exclude the whole constructor
# }


class pipe_isolation:
    """
    Description
    ----------  
    This class contains the function `pipe_heat` which is directly referenced by the calling code.    
   
    Depending on the defined conditions of the system, `pipe_heat` will call one of the following functions:
    `pipe_Q_def`, `pipe_Tw_def`,`pipe_h_ext`, `pipe_insulated`.    
    
    Additionally, the functions `laminar_flow`, `turbulent_flow`, `dP_Pipe`, `find_Tw`, `k_pipe`, and `h_ext_`
    are used throughout the script to determine the thermal and fluid properties of the system. 
    
    Assumptions
    ---------- 
    - System is at steady state
    - 1 dimensional heat transfer
    - Constant material properties
    - Constant wall temperature
    - Uniform wall thickness
    """

    def __init__(self, k, OD, T_ext=293 * ureg.K):  
         self.k = k
         self.OD = OD
         self.T_ext = T_ext 


def laminar_flow(Re_, Pr_, L_ID):
    """
    Non dimentional calculation of the Nusselt and fiction factor in pipe in laminar flow  
    Section 5.2.4 of Nellis and Klein (2020)
    """
    # Verify the input conditions     
    if Pr_ < 0.1:
        raise ValueError(f'Prandtl number (Pr) must be > 0.1. The value is {Pr}')
        
    # Calculate Graetz number and Inverse Graetz number verification
    SGZ = L_ID / Re_
    GZ = L_ID / (Re_ * Pr_)
    if GZ < 1e-6:
        raise ValueError(f'Inverse Graetz number (GZ) must be > 1e-6. The value is {GZ}')
         
    # Calculate friction factor (f)
    f = 4 * (3.44 / sqrt(SGZ) + (1.25 / (4 * SGZ) + 16 - 3.44 / sqrt(SGZ)) / (1 + 0.00021 * SGZ**(-2))) / Re_
    
    # Calculate Nusselt numbers: temperature constant and flux constant
    Nu_T = ((5.001 / GZ**1.119 + 136.0)**0.2978 - 0.6628) / tanh(2.444 * SGZ**(1 / 6) * (1 + 0.565 * SGZ**(1 / 3)))
    Nu_Q = ((6.562 / GZ**1.137 + 220.4)**0.2932 - 0.5003) / tanh(2.530 * SGZ**(1 / 6) * (1 + 0.639 * SGZ**(1 / 3)))
    
    return Nu_T, Nu_Q, f

def turbulent_flow(Re_, Pr_, L_ID, eps):
    """
    Non dimentional calculations of the Nusselt and fiction factor in pipe in turbulent flow following Section 5.2.3 of Nellis and Klein (2020)
    """
    ##Verify the input conditions
    if Pr_ < 0.004 or Pr_ > 2000:
        raise ValueError(f'Prandtl number (Pr) must be between 0.004 and 2000. The value is {Pr}')
    if L_ID <= 1:  
        if L_ID < 0:  ###not inferior to zero - make no sense
            raise ValueError('L/ID ratio < 0. Not possible')
        print('L/ID ratio should be > 1. The value is {L_ID}')
        L_ID = 1
        
    # Friction Factor 
    if eps > 1e-5:
        #Offor & Alabi, Advances in Chemical Engineering and Science, 2016
        friction = (-2 * log10(eps / 3.71 - 1.975 / Re_ * log((eps / 3.93)**1.092 + 7.627 / (Re_ + 395.9))))**(-2)
    else:
        #Li & Seem correlation, A New Explicity Equation for Accurate Friction Factor Calculation for Smooth Tubes, 2011
        friction = (-0.001570232 / log(Re_) + 0.394203137 / log(Re_)**2 + 2.534153311 / log(Re_)**3) * 4
        
    # Nusselt, Gnielinski, Int. Chem. Eng., 1976
    Nusselt = ((friction / 8) * (Re_ - 1000) * Pr_) / (1 + 12.7 * sqrt(friction / 8) * (Pr_**(2 / 3) - 1))
    
    # Correct Nusselt number for low Prandtl numbers, Notter & Sleicher, Chem. Eng. Sci., 1972
    if Pr_ < 0.5:
        Nusselt_lp = 4.8 + 0.0156 * Re_ ** 0.85 * Pr_ ** 0.93
        if Pr_ < 0.1:
            Nusselt = Nusselt_lp
        else:
            Nusselt = Nusselt_lp + (Pr_ - 0.1) * (Nusselt - Nusselt_lp) / 0.4   
            
    # Correct f and Nusselt for developing flow
    f = friction * (1 + (1 / L_ID)**0.7)
    Nu = Nusselt * (1 + (1 / L_ID)**0.7)
    
    return Nu, f

def dP_Pipe(m_dot, fluid, pipe):
    """
    Calculate pressure drop for flow with heat transfer at the surface of the pipe.   
    Section 5.2.3 and 5.2.4 of Nellis and Klein (2020)
    Parameters
    ----------
    `fluid` : ThermState | Inlet fluid conditions   
    `m_dot` : Quantity {mass: 1, time: -1} | mass flow rate     
    `pipe` : Pipe | defines the pipe characteristics 
    Returns
    -------
    `dP`: Quantity {length: -1, mass: 1, time: -2} | Pressure drop   
    `h_T`, `h_Q`: Quantity: {mass : 1, temperature : -1, time : -3}
         | Heat transfer coefficients
     -------
    """

    # Calculate fluid pameters
    Re_ = Re(fluid, m_dot, pipe.ID, pipe.area)
    L_ID = pipe.L.m_as(ureg.m)/pipe.ID.m_as(ureg.m)
    eps = (pipe.eps/pipe.ID)
    w = velocity(fluid, m_dot, pipe.area)
    Pr = fluid.Prandtl
    
    #Verify two phase flow and Chockedflow
    if (fluid.phase == 0 or fluid.phase == 6) and fluid.Q < 0.9: 
        Phase = 'liquid or two-phase'
    else:
        if Mach(fluid, w) > 1/(fluid.gamma):
            raise ChokedFlow(' Reduce hydraulic resistance or mass flow.')

    # Check Flow conditions
    if Re_ < 0.001 or Re_ > 5e7: 
        raise ValueError(f'Reynolds number (Re) must be between 0.001 and 5E7. The value is {Re_}')
    if eps < 0 or eps > 0.05:
        raise ValueError(f'Relative roughness (eps) should be between 0 and 0.05. The value is {eps}')

    if Re_ > 3000:  # Turbulent flow
        Nu_T, f = turbulent_flow(Re_, Pr, L_ID, eps)
        Nu_Q = Nu_T
        
    elif Re_ < 2300:  # Laminar flow
        Nu_T, Nu_Q, f = laminar_flow(Re_, Pr, L_ID)
        
    else:  # Transitional flow (Re between 2300 and 3000)
        Nu_T_turbulent, f_turbulent = turbulent_flow(3000, Pr, L_ID, eps)
        Nu_lam_T, Nu_lam_Q, f_lam = laminar_flow(2300, Pr, L_ID)
        
        # Interpolate between laminar and turbulent values
        alpha = (Re_ - 2300) / (3000 - 2300)
        Nu_T = Nu_lam_T + alpha * (Nu_T_turbulent - Nu_lam_T)
        Nu_Q = Nu_lam_Q + alpha * (Nu_T_turbulent - Nu_lam_Q)
        f = f_lam + alpha * (f_turbulent - f_lam)
     
    # Calculate pressure drop
    dP = dP_Darcy(f*L_ID, fluid.Dmass, w)    
    
    # Calculate heat transfer coefficient: temperature and heat flux constant 
    h_T = heat_trans_coef(fluid, Nu_T, pipe.ID)
    h_Q = heat_trans_coef(fluid, Nu_Q, pipe.ID)
    
    return dP.to(ureg.pascal), h_T, h_Q

def find_Tw(x, T_avg, pipe, h_coeff, m_dot, condition): #add parameter conditions
  
    """
    Description
    ---------- 
    This function is used to find the wall temperature of a pipe. 
    The system can either have a defined applied heat flux (`pipe_Q_def`), pipe outer wall temperature (`pipe_Tw_def`), 
    external fluid heat transfer coefficient (`pipe_h_ext`), or pipe insulation (`pipe_insulated`).  
    <div style="text-align: center;">
        <img src="find_Tw.jpg" alt="find_Tw" width="600" />
    </div> 
    Depending on the inputs, either the inner or outer wall temperature is calculated. The heat flux, $q"$ is calculated 
    from the general convection equation as a function of the unknown wall temperature shown below in Equation #. Here $h$ is the heat transfer
    coefficient, $T_{avg}$ is the average fluid temperature, and $Tw_i$ and $Tw_o$ are the inner and outer wall temperatures respectively.      
    $$ q"_{conv} = h * (Tw_i – T_{avg}) $$ 
    Similarly, the heat flux is also calculated as a function of the unknown wall temperature from the conduction equation.  
    $$ q"_{cond} = (Tw_o – Tw_i)/ R_L $$ 
    $ R_L $ is the thermal resistance of the wall defined below in Equation # where $k$ is the thermal resistance of the wall, $L$ is the length of the pipe, 
    and $D_o$ and $D_i$ are the outer and inner diameters of the pipe respectively. 
    $$ R_L = log(D_o/D_i) / (2 * pi * L * k) $$ 
    These two expressions for heat flux are set equal to each other and squared to create a quadratic.     
    $$ (q"_{conv} - q"_{cond})^2 = 0 $$ 
    The minimize function is used to find the solution of the quadratic: the unknown wall temperature.   
    Parameters
    ----------
  
    `dT` : Quantity {temperature: 1}
        | temperature difference   
    `fluid` : ThermState
        | Inlet fluid conditions   
    `fluid_external` : Thermstate
        | External fluid conditions   
    `h_coeff`: Quantity {mass : 1, temperature : -1, time : -3}
        | heat transfer coefficient: h_T or h_Q   
    `h_ext`: Quantity {mass : 1, temperature : -1, time : -3}
        | heat transfer coefficient for external fluid   
    `k` : Quantity {length: 1, mass: 1, temperature: -1, time: -3}
        | material thermal conductivity   
    `m_dot` : Quantity { mass: 1, time: -1}
        | mass flow rate       
    `pipe` : Pipe
        | defining the pipe characteristics         
    `Q_cond` : Quantity {length: 2, mass: 1, time: -2}
        | heat rate calculated from thermal resistance equation    
    `Q_conv` : Quantity {length: 2, mass: 1, time: -2}
        | heat rate calculated from convection equation         
    `Q_def` : Quantity {mass: 1, time: -3}
        | Heat load reaching the fluid     
    `Tw_i`: Quantity {temperature: 1}
        | inside temperature of wall     
    `Tw_o`: Quantity {temperature: 1}
        | outside temperature of wall         
    Returns
    -------
    ``(Q_conv - Q_cond) ** 2`` :
            quadratic expression that computes wall temperature when the minimum is solved
     -------
    """
    
    if condition == 1:
        #For a system with defined heat load: pipe_Q_def
        Tw_i = T_avg + pipe.Q_def/h_coeff * (pipe.OD / pipe.ID)
        Tw_o =  x * ureg.K
        
    elif condition == 2:
        #For a system with defined external temperature: pipe_Tw_def
        Tw_i = x * ureg.K
        Tw_o = pipe.Tw_def
        
    elif condition == 3:
        #For a system with defined external heat transfer coeff: pipe_h_ext
        fluid_external = ThermState('air', T= pipe.T_ext, P=1 * ureg.bar) #to do: improve structure 
        h_ext = h_ext_(fluid_external, pipe, x * ureg.K)     
        
        Tw_i = (h_ext * pipe.OD * (pipe.T_ext - x * ureg.K) / h_coeff / pipe.ID) + T_avg
        Tw_i = max(min(Tw_i, pipe.T_ext), T_avg)
        Tw_o =  x * ureg.K
        
    elif condition == 4: 
        #For a defined insulated system: pipe_insulated
        Tw_i = (pipe.isolation.k * (pipe.isolation.T_ext - x * ureg.K) / h_coeff / pipe.ID / log(pipe.isolation.OD / pipe.OD)) + T_avg
        Tw_i = max(min(Tw_i, pipe.isolation.T_ext), T_avg)
        Tw_o =  x * ureg.K
    else:
        raise ValueError("Insufficient or invalid parameters provided.")                
    
    k = k_pipe(pipe, Tw_o, Tw_i)
    Q_cond = conduction_cyl(pipe.ID.to(ureg.m), pipe.OD.to(ureg.m), pipe.L.to(ureg.m), k, (Tw_o - Tw_i))
    Q_conv = -h_coeff * (T_avg - Tw_i) * pipe.ID.to(ureg.m) * pipe.L.to(ureg.m) * 3.14  
    
    return (Q_conv - Q_cond).m_as(ureg.W) ** 2  


def pipe_Q_def(fluid, pipe, m_dot, dP, h_Q):
    """
    Description
    ----------    
    This function is used for systems with a defined constant heat flux. The set-up of the function is displayed below in Figure #. 
    <div style="text-align: center;">
        <img src="pipe_Q_def.jpg" alt="pipe_Q_def" width="550" />  
    </div>     
    The heat flux is used to calculate the fluid’s change of enthalpy. This is done using Equation # shown below where $dH$ is the 
    change in enthalpy, $q"$ is the defined heat flux, $S_i$ is the pipe's inner surface area, and $m_{dot}$ is the mass flow rate.  
    $$ dH = q" * S_i/m_{dot} $$
    Using the pressure drop calculated from dP_pipe, the outlet conditions are calculated. The average temperature of the fluid is determined from the inlet and
    outlet temperatures and assumes a linear temperature gradient. This average temperature is used to calculate the average inner wall temperature using the general 
    convection equation shown below in Equation #. 
    $$ Tw_i = Q/(h_Q * S_i) + T_{avg}  $$
    In the above expression, $Q$ is the heat load, $h_Q$ is the heat transfer coefficient calculated considering a constant heat flux, 
    and $T_{avg}$ is the avaerage temperatue of the fluid. Finally, the outer wall temperature is calculated using function `find_Tw`.
    Parameters
    ----------
    `dH`: Quantity {length: 2, time: -2} 
        | specific enthalpy of fluid    
    `fluid` : ThermState
        | inlet fluid conditions   
    `fluid_outlet` : Thermstate
        | outlet fluid conditions   
    `m_dot` : Quantity { mass: 1, time: -1}
        | mass flow rate          
    `pipe` : Pipe
        | Defining the pipe characteristics    
    `Q_def` : Quantity {mass: 1, time: -3}
        | heat load reaching the fluid   
    `T_avg` : Quantity {temperature: 1}
        | average fluid temperature 
    Returns
    -------
    `Tw_i`, `Tw_o` : Quantity {temperature: 1}
        | inside temperature of the wall, outside temperature of the wall
    """    
    #Calculate downstream conditions
    fluid_outlet = fluid.copy()
    dH = (pipe.Q_def * pipe.OD.to(ureg.m) * pipe.L.to(ureg.m) * 3.14) / m_dot #should be the outer diameter, could explain issues
    fluid_outlet.update('P', fluid.P - dP, 'Hmass' , fluid.Hmass + dH.to(ureg.J/ureg.kg)) 
    
    ##Calculate the average temperature of the fluid inside the component
    T_avg = (fluid.T + fluid_outlet.T)/2   
    
    #Calculate Tw_i and Tw_o: minimum of the quadratic find_Tw
    Tw_i = T_avg + pipe.Q_def / h_Q * (pipe.OD / pipe.ID)
    
    # Assign bounds for minimize function to operate within
    dT = log(pipe.OD/pipe.ID)*(pipe.Q_def*pipe.OD)/(2*k_pipe(pipe,Tw_i))
    
    # Calculate Tw_o: minimum of the quadratic find_Tw
    condition = 1 # condition to designate Q_def in find_tw
    if pipe.Q_def == 0*ureg.W/ureg.m**2 :
        Tw_o = T_avg
    else:
        # Tw_o = minimize(find_Tw, x0=T_avg.m_as(ureg.K) + 1, args=(T_avg, pipe, h_Q, m_dot, condition), bounds=bounds).x[0] * ureg.K
        Tw_o = root_scalar(find_Tw, x0 = (Tw_i + dT/2).m_as(ureg.K), x1 = (Tw_i + dT).m_as(ureg.K), args=(T_avg, pipe, h_Q, m_dot, condition)).root * ureg.K

    return Tw_i, Tw_o  


# Define main functions
def pipe_Tw_def(fluid, pipe, m_dot, dP, h_T):
    """
    Description
    ----------    
    This function is used for systems with a constant pipe outer wall temperature. 
    The set-up is shown below in Figure # with the inputs highlighted. 
    <div style="text-align: center;">
        <img src="pipe_Tw_def.jpg" alt="pipe_Tw_def" width="600" />   
    </div>         
    The inner wall temperature is calculated in `find_Tw` from the average temperature which is initially assigned to a reference value.
    The inner wall temperature is used in Equation # described in `pipe_Q_def` to calculate the change in enthalpy of the fluid.
    The outlet pressure is calculated in `dP_Pipe`. Additionally, the outlet pressure is calculated in `dP_pipe`. 
    Knowing these fluid conditions, the outlet temperature of the fluid is determined.
    The average temperature is then re-calculated from the inlet and outlet fluid temperatures using Equation #.  
    $$ T_{avg} = (T_{inlet} + T_{outlet})/2 $$
    This value replaces the reference value, and the code loops until the reference temperature and the the calculated average temperature converge.
    Once $T_{avg}$ is known, the heat flux is calculated from the general convection equation shown below in Equation # where $h_T$ is the heat transfer coefficient calculated in `dP_Pipe`. 
    $$q” = -h_T * (T_{avg}-Tw_i) $$ 
    Parameters
    ----------
    `dH` : Quantity {length: 2, time: -2}
        | specific enthalpy of fluid   
    `dT` : Quantity {temperature: 1}
        | temperature difference   
    `fluid` : ThermState
       | inlet fluid conditions   
    `fluid_out` : Thermstate
       | outlet fluid conditions   
    `h_T` : Quantity : {mass : 1, temperature : -1, time : -3}
        | heat transfer coefficient    
    `m_dot` : Quantity { mass: 1, time: -1}
        | mass flow rate               
    `pipe` : Pipe
        | define the pipe characteristics    
    `T_out` : Quantity {temperature: 1}
        | temperature of outlet fluid     
    Returns
    -------
    `Tw_i` and `Tw_o` : Quantity {temperature: 1}
        | inside temperature of the wall, outside temperature of the wall   
    `Q` : Quantity { mass: 1, time: -3}
        | heat load reaching the fluid
     ---------
    """
    #### Initial conditions and parameters
    H = fluid.Hmass
    fluid_out = fluid.copy()
    T_avg = fluid.T
    Tw_o = pipe.Tw_def
    res = 1
    j = 0
    dT = 'none'
   
    while res>0.0001:  
        
        # Limits search range for Tw_o
        if T_avg < Tw_o:
            bracket = (T_avg.m_as(ureg.K), Tw_o.m_as(ureg.K)) 
        else:
            bracket = (Tw_o.m_as(ureg.K), T_avg.m_as(ureg.K))
            
        # Calculate Tw_i:  minimum of the quadratic find_Tw
        condition = 2 # condition to designate Tw_def in find_tw

        Tw_i = minimize(find_Tw,  x0 = T_avg.m_as(ureg.K), args = (T_avg, pipe, h_T, m_dot, condition), bounds=[bracket]).x[0] *ureg.K        
        
        ### Calculate downstream fluid conditions
        dT = T_avg - Tw_i
        dH = - h_T * dT * pipe.ID * pipe.L * 3.14 / m_dot

        fluid_out.update('P', fluid.P - dP,'Hmass', H + dH.to(ureg.J/ureg.kg))
        T_out = fluid_out.T   
 
        
        ###Check convergence of T_average               
        T_avg_new = (fluid.T + T_out)/2
        res = ((T_avg_new - T_avg)**2 / (T_out - fluid.T)**2)

        ##Update average temperature
        T_avg = T_avg_new

        
        ### Eliminate nonphysical solutions         
        if (fluid.T < Tw_o and T_out > Tw_o) or (fluid.T > Tw_o and T_out < Tw_o):
            if j > 0:
                raise Exception('the pipe is too long')
            j += 1
            T_avg = (fluid.T + Tw_o) / 2

        ### Calculate heat flux 
        Q = (- h_T * dT ).to(ureg.W/ureg.m ** 2)       
        
    return Tw_i, Tw_o, Q


def pipe_h_ext(fluid, pipe, m_dot, dP, h_T): 
    """
    Description
    ----------    
    This function is used for systems with defined external conditions. Either the heat transfer coefficient of the external fluid can be defined, or 
    methods of heat transfer to be considered can be specified. This is computed and explained in `heated_pipe`.    
    The set-up for this function is displayed below in Figure # with the input variables highlighted. 
    <div style="text-align: center;">
        <img src="pipe_h_ext.jpg" alt="pipe_h_ext" width="600" />  
    </div>     
    The outer wall temperature, $Tw_o$, is calculated in `find_Tw` from the average temperature, $T_{avg}$, which is initially assigned to a reference value.
    The expression for the heat flux due to convection of the external fluid and the outer wall is defined below in Equation # where $h_{ext}$ is the external 
    heat transfer coefficient, $D_o$ is the outer diameter, $D_i$ is the inner diameter, and $T_{ext}$ is the external fluid temperature. 
    The external fluid heat transfer coefficient is determined in the function `h_ext_`.
    $$ q" = (T_{ext}-Tw_o) * h_{ext} * S_o $$
    The heat flux can also be calculated from the convection between the internal fluid and the inner wall defined below in Equation #. 
    $$ q" = (Tw_i - T_{avg}) * h_T * S_i $$
    Combining and similifying these equations, an expression for the inner wall temperature, Equation #, is found.  
    $$Tw_i = [h_{ext} * D_o * (T_{ext} - Tw_o) / (h_T * D_i)] + T_{avg}$$
    The inner wall temperature is calculated and used in Equation # described in `pipe_Q_def` to calculate the change in enthalpy of the fluid.
    Additionally, the outlet pressure is calculated in `dP_Pipe`. 
    Knowing these fluid conditions, the outlet temperature of the fluid is determined. The average temperature is then re-calculated from the inlet and outlet fluid 
    conditions using Equation # explained above in `pipe_Tw_def`. This value replaces the reference value, and the code loops until the reference temperature and the the calculated average temperature converge. 
    Once $T_{avg}$ is known, the heat flux is calculated from Equation #.
    Parameters
    ----------
     `dH` : Quantity {length: 2, time: -2}
        | specific enthalpy of fluid   
    `dT` : Quantity {temperature: 1}
        | temperature difference   
    `fluid` : ThermState
       | inlet fluid conditions   
     `fluid_out` : Thermstate
       |  outlet fluid conditions   
     `fluid_external` : Thermstate
         | external fluid conditions   
    `h_ext` : Quantity : {mass : 1, temperature : -1, time : -3}
        | transfer coefficient of external fluid   
    `m_dot` : Quantity { mass: 1, time: -1}
        | mass flow rate           
    `pipe` : Pipe
        | define the pipe characteristics    
    `T_avg` : Quantity {temperature: 1}
        | average temperature of the fluid   
    `T_ext` : Quantity {temperature: 1}
        | average temperature of the external fluid          
    `T_out` : Quantity {temperature: 1}
        | temperature of oulet fluid  
    Returns
    ---------
     `Tw_i` and `Tw_o` : Quantity {temperature: 1}
         inside temperature of the wall, outside temperature of the wall   
    `Q` : Quantity { mass: 1, time: -3}
         heat load reaching the fluid
     --------
    """
    ### Initial conditions and parameters
    H = fluid.Hmass
    fluid_out = fluid.copy()
    T_avg = fluid.T
    res = 1
    j = 0
    
    while res>0.0001:  
        
        ##Define external fluid
        fluid_external = ThermState('air', T= pipe.T_ext, P=1 * ureg.bar) 
        
        # Limits search range for Tw_o
        if T_avg < pipe.T_ext:   
            bracket = [(T_avg.m_as(ureg.K)+0.0001, pipe.T_ext.m_as(ureg.K)-0.0001)]   
        else:
            bracket = [(pipe.T_ext.m_as(ureg.K)+0.0001, T_avg.m_as(ureg.K)-0.0001)]       
        
        #Calculate Tw_o: minimum of the quadratic find_Tw_o
        condition = 3
        Tw_o = minimize(find_Tw, x0=(fluid_external.T + T_avg).m_as(ureg.K)/ 2, args = (T_avg, pipe, h_T, m_dot, condition), bounds=bracket).x[0] * ureg.K                                                  
        
        # Caclulate external heat transfer coefficient for system: calculates h_ext for system with h_type defined otherwise uses defined h_ext
        h_ext = h_ext_(fluid_external, pipe, Tw_o)
        
        # Calculatue inner wall temperature
        Tw_i = ( h_ext * pipe.OD * (pipe.T_ext - Tw_o) / h_T / pipe.ID  ) + T_avg
        
        ###Calculate downstream flow conditions
        dT = T_avg - Tw_i
        dH = - h_T * dT * pipe.ID * pipe.L * 3.14 / m_dot
        fluid_out.update('P', fluid.P - dP,'Hmass', H + dH.to(ureg.J/ureg.kg))
        T_out = fluid_out.T 

        ###Check convergence of T_average         
        T_avg_new = (fluid.T + T_out)/2
        #res = ((T_avg_new - T_avg)**2 / (T_out - fluid.T)**2)
        res = 0.00001
        ##Update average temperature
        T_avg = T_avg_new
    
        ###Eliminate nonphysical solutions         
        if (fluid.T < Tw_o and T_out > Tw_o) or (fluid.T > Tw_o and T_out < Tw_o):
            if j > 0:
                raise Exception('the pipe is too long')
            j += 1
            T_avg = (fluid.T + Tw_o) / 2
       
        ###Calculate heat flux  
        Q = (- h_T * dT ).to(ureg.W/ureg.m ** 2)
    
        return Tw_i, Tw_o, Q

def pipe_insulated(fluid, pipe, m_dot, dP, h_T): 
    """
    Description
    ----------    
    This function is used for piping systems with insulation. The insulation must have a defined thermal conductivity and outer diameter. 
    The external temperature of the insulation is assumed to be $293K$ if not specifically defined.   
    The set-up of the function is shown below in Figure # where the variable inputs to the function are highlighted.
    <div style="text-align: center;">
        <img src="pipe_insulated.jpg" alt="pipe_insulated" width="600" />
    </div>
    The outer wall temperature, $Tw_o$, is calculated in `find_Tw` from the average temperature, $T_{avg}$ , which is initially assigned to a reference value.
    The expression for the heat flux due to convection of the internal fluid and the inner wall is defined below in Equation # where $h_{ext}$ is the external 
    heat transfer coefficient, $D_o$ is the outer diameter, $ D_i $ is the inner diameter, and $T_{ext}$ is the external fluid temperature.
    $$ q" = (Tw_i - T_{avg}) * h_T * S_i $$
    The heat flux can also be calculated from the conduction between the insulation with the outer wall of the pipe defined below in Equation # where $R_L$ is the thermal resistance of the wall. 
    $$ q" = (T_{ext}-Tw_o) / R_L $$
    Combining and similifying these equations, an expression for the inner wall temperature, Equation #, is found.  
    $$ Tw_i = ((k_{ins}* (T_{ext} - Tw_o) )/ (h_T * D_i * log(D_o/D_i)) +T_{avg} $$ 
    The inner wall temperature is calculated and used in Equation # described in `pipe_Q_def` to calculate the change in enthalpy of the fluid.
    Additionally, the outlet pressure is calculated in `dP_pipe`. 
    Knowing these fluid conditions, the outlet temperature of the fluid is determined. The average temperature is then re-calculated from the inlet and outlet fluid 
    conditions using Equation # explained above in `pipe_Tw_def`. This value replaces the reference value, and the code loops until the reference temperature and the the calculated average temperature converge. 
    Once $T_{avg}$ is known, the heat flux is calculated from Equation #.
    Parameters
    ----------
     `dH` : Quantity {length: 2, time: -2}
        | specific enthalpy of fluid   
    `dT` : Quantity {temperature: 1}
        | temperature difference   
    `fluid` : ThermState
        | inlet fluid conditions   
    `fluid_out` : Thermstate
        | outlet fluid conditions   
    `m_dot` : Quantity { mass: 1, time: -1}
        | mass flow rate           
    `pipe` : Pipe
        | define the pipe characteristics    
    `T_avg` : Quantity {temperature: 1}
        | average temperature of the fluid  
    `T_ext` : Quantity {temperature: 1}
        | average temperature of insulation's external wall    
    `T_out` : Quantity {temperature: 1}
        | temperature of outlet fluid  
    Returns
    ----------
     `Tw_i` and `Tw_o` : Quantity {temperature: 1}
         | inside temperature of the wall, outside temperature of the wall   
     `Q` : Quantity { mass: 1, time: -3}
        | heat load reaching the fluid
     ----------
    """
    ###Initial conditions and parameters
    H = fluid.Hmass
    fluid_out = fluid.copy()
    T_avg = fluid.T
    res = 1
    j = 0
    
    while res>0.0001:   
        # Limits search range for Tw_o       
        if T_avg < pipe.isolation.T_ext:  
            bracket = [(T_avg.m_as(ureg.K)+0.0001, pipe.isolation.T_ext.m_as(ureg.K)-0.0001)]  
        else:
            bracket = [(pipe.isolation.T_ext.m_as(ureg.K)+0.0001, T_avg.m_as(ureg.K)-0.0001)]
         
        ###Calculate Tw_i and Tw_o: minimum of the quadratic find_Tw
        condition = 4
        Tw_o = minimize(find_Tw, x0=pipe.isolation.T_ext.m_as(ureg.K) - (pipe.isolation.T_ext.m_as(ureg.K) - T_avg.m_as(ureg.K)), args = (T_avg, pipe, h_T, m_dot, condition), bounds=bracket).x[0] * ureg.K 
        Tw_i = (( pipe.isolation.k * (pipe.isolation.T_ext - Tw_o) / h_T / pipe.ID / log(pipe.isolation.OD/pipe.OD) ) + T_avg)
        
        ###Calculate downstream flow conditions
        dT = T_avg - Tw_i
        dH = - h_T * dT * pipe.ID * pipe.L * 3.14 / m_dot
        fluid_out.update('P', fluid.P - dP,'Hmass', H + dH.to(ureg.J/ureg.kg))
        T_out = fluid_out.T 

        ###Check convergence of T_average       
        T_avg_new = (fluid.T + T_out)/2
        res = ((T_avg_new - T_avg)**2 / (T_out - fluid.T)**2)
        
        ###Update T_avg value  
        T_avg = T_avg_new
        
        ###Eliminate nonphysical solutions
        if (fluid.T < Tw_o and T_out > Tw_o) or (fluid.T > Tw_o and T_out < Tw_o):
            if j > 0:
                raise Exception('the pipe is too long')
            j += 1
            T_avg = (fluid.T + Tw_o) / 2
        
        ###Calculate heat flux 
        Q = (- h_T * dT ).to(ureg.W/ureg.m ** 2)

        return Tw_i, Tw_o, Q


def pipe_heat(pipe, fluid, m_dot):
    """   
    Determine the heated status of the piping component with one of the following systems:   
        
        1. Defined applied heat flux    
        2. Constant outerwall temperature
        3. Defined properties for the external fluid 
        4. Insulation   
           
    If no parameters are specified then `pipe_Q_def` is run using a heat load of $ Q = 0  W/m^2 $. 
        
    """
    ### Calculate pressure drop and heat transfer coefficient
    dP, h_T, h_Q = dP_Pipe(m_dot, fluid, pipe)  
    
    ###heat flux defined on the wall of the pipe
    if hasattr(pipe, 'Q_def') and pipe.Q_def != None :
        try: 
            pipe.Q_def.m_as(ureg.W / ureg.m ** 2)
        except:
            raise ValueError(f"the Q_def is not properly defined in component {pipe}" )
        Tw_i, Tw_o = pipe_Q_def(fluid, pipe, m_dot, dP, h_Q)
        Q = pipe.Q_def
        
    ###Temperature defined on the wall of the pipe
    elif hasattr(pipe, 'Tw_def') and pipe.Tw_def != None :
        try: 
            pipe.Tw_def.m_as(ureg.K)
        except:
            raise ValueError(f"the Tw_def is not properly defined in component {pipe}" )
        Tw_i, Tw_o, Q = pipe_Tw_def(fluid, pipe, m_dot, dP, h_Q)

    ### Heat transfer or ambient/external temperature defined on the wall of the pipe
    elif (hasattr(pipe, 'h_ext') and pipe.h_ext is not None) or (hasattr(pipe, 'considerations') and pipe.considerations is not None):
      
        if hasattr(pipe, 'h_ext') and pipe.h_ext is not None:
            try:
                pipe.h_ext.m_as(ureg.W / ureg.m ** 2 / ureg.K)
            except AttributeError:
                raise ValueError(f"the h_ext is not properly defined in component {pipe}")
                
        # Ensure T_ext is defined, or set it to a default value
        if not hasattr(pipe, 'T_ext') or pipe.T_ext is None:
            pipe.T_ext = 293 * ureg.K  # Default to 293K if T_ext is not defined
        else:
            try:
                pipe.T_ext.m_as(ureg.K)
            except AttributeError:
                raise ValueError(f"The T_ext is not properly defined in component {pipe}")

        # Check and define considerations-related parameters
        if hasattr(pipe, 'considerations') and pipe.considerations is not None:
            if not hasattr(pipe, 'T_ext') or pipe.T_ext is None:
                pipe.T_ext = 293 * ureg.K  # Ensure T_ext is defined when using considerations
            else:
                try:
                    pipe.T_ext.m_as(ureg.K)
                except AttributeError:
                    raise ValueError(f"The T_ext is not properly defined in component {pipe}")
              
                try:
                    pipe.safety_fact
                except:
                    pipe.safety_fact = 1
        
        Tw_i, Tw_o, Q = pipe_h_ext(fluid, pipe, m_dot, dP, h_T)  

    ###Isolation on the external of the pipe 
    elif hasattr(pipe, 'isolation') and pipe.isolation != None :
        try: 
            pipe.isolation.k.m_as(ureg.W / ureg.m / ureg.K)
            pipe.isolation.OD.m_as(ureg.m)
        except:
            raise ValueError(f"the isolation (k, OD) is not properly defined in component {pipe}" )
        Tw_i, Tw_o, Q = pipe_insulated(fluid, pipe, m_dot, dP, h_T)
                
    ###Other
    else: 
        pipe.Q_def = Q = 0 * ureg.W/ ureg.m ** 2
        Tw_i, Tw_o = pipe_Q_def(fluid, pipe, m_dot, dP, h_Q)
    
    return Tw_i.to(ureg.K), Tw_o.to(ureg.K), dP.to(ureg.bar), Q.to(ureg.W/ ureg.m ** 2)
        

def k_pipe(pipe, T_wall, T_ext = 293 * ureg.K):     ### you should add the possibility to define a table of values to do that (for materials not in the database)
    """This function assigns the thermal conductivity, k.    
    Either the conductivity is specifically defined or it is calculated accourding to the material and the input temperatures (typically the inner 
    and outer wall temperatures) using the NIST property tables.
    If no material or outer wall temperature is defined, stainless steel 304 and ambient temperature (293K) is assumed receptively.
    """
    try:
        # Check if the 'pipe' object has the attribute 'k'
        if hasattr(pipe, 'k'):
            k = pipe.k.m_as(ureg.W / ureg.K / ureg.m) * ureg.W / ureg.K / ureg.m
        else:
            raise AttributeError("Pipe object has no attribute 'k'")
    except AttributeError:
        # Handle the case where 'k' is not an attribute of 'pipe'
        try:
            mat = pipe.material
        except AttributeError:
            # If 'pipe' has no 'material' attribute, default to a material
            if isinstance(pipe, CopperTube):
                mat = Material.OFHC
            else:
                mat = Material.SS304
       ### Determine the thermal conductivity using the 'nist_property' function
        if np.abs((T_wall - T_ext).m_as(ureg.K)) < 0.001:
            k = nist_property(mat, Property.TC, T1=T_wall)
        else:
            if T_wall < T_ext:
                k = nist_property(mat, Property.TC, T1=T_wall, T2=T_ext)
            else:
                k = nist_property(mat, Property.TC, T1=T_ext, T2=T_wall)
    return k.to(ureg.W / ureg.K / ureg.m)

# def h_ext_(fluid, pipe, T_wall):  ### you should add the possibility to define a table of values to do that
#     """Determine the heat transfer coefficient of the external fluid
    
#     """
#     try:
#         h = pipe.h_ext.m_as(ureg.W / ureg.K / ureg.m ** 2) * ureg.W / ureg.K / ureg.m ** 2
#     except:
#         try:
#             type = pipe.h_type
#         except:   
#             raise Exception('You should define an external heat transfer type pipe.h_type') #or define type = 1
#         if type >= 1: #Convection considered
            
#             #Calculate Rayleigh and Prandtl numbers
#             Ra_ = Ra(fluid, T_wall, pipe.OD)
#             Pr_ =  (fluid)
            
#             #Determine orientation of the pipe      
#             try:
#                 orientation = pipe.orientation
#             except:
#                 orientation = None
#             #Calculatue Nusselt numbers  
#             if orientation == 'vertical':        
#                 Nu_ = Nu_vcyl(Pr_, Ra_, pipe.OD, pipe.L)
#             elif orientation == 'horizontal':
#                 Nu_ = Nu_hcyl(Pr_, Ra_)
#             else:
#                 #Use maximum Nusselt if orientation is not defined
#                 Nu_ = max (Nu_vcyl(Pr_, Ra_, pipe.OD, pipe.L), Nu_hcyl(Pr_, Ra_))
            
#             #Calculate heat transfer coefficient of external fluid
#             h = heat_trans_coef(fluid, Nu_, pipe.OD)
        
#         if type == 2: #Convection and radiation considered
            
#             #Determine emissivity of material
#             try:
#                 epsilon = pipe.epsilon
#             except:
#                 #Assume material is polished steel if not defined
#                 epsilon = 0.075 
            
#             #Calculate radiation heat transfer coefficient
#             sigma = 5.670373e-8 * ureg.W/ureg.m ** 2 / ureg.K ** 4
#             h_rad = epsilon * sigma * (fluid.T ** 4 - T_wall ** 4) / (fluid.T - T_wall) 
            
#             #Calculate total heat transfer coefficient 
#             h = h + h_rad

#         if type == 3: ###including Rad ice and h_ice specific in the problem: to do 
#             print('to do')
#         else:
#             raise ValueError("Insufficient or invalid parameters provided. Please provide an external heat transfer coefficient or an h_type")   
   
#     return h.to(ureg.W / ureg.K / ureg.m ** 2)

def h_ext_(fluid, pipe, T_wall, considerations=None, humidity=0.5): ### you should add the possibility to define a table of values to do that
    """This function determines the heat transfer coefficient of the external fluid surrounding the pipe.    
    If an external heat transfer coefficient is not defined, then it is calculted assuming an external fluid of air at 293 K and 1 bar. 
    Specific heat transfer conditions can be specified in the variable pipe.considerations. 
    Considerations can include:     
        
        1. Convection     
        2. Radiation  
        3. Icing    
        
    For considerations with radiation, if an emissivity is not defined the system will assume polished steel (emissivity = 0.075). 
    """
    if isinstance(considerations, str):
        considerations = (considerations,)

    # Set default considerations if not already set
    if not hasattr(pipe, 'considerations'):
        setattr(pipe, 'considerations', ('convection', 'radiation', 'icing'))
    else:
        # Ensure pipe.considerations is a tuple if set externally
        if isinstance(pipe.considerations, str):
            pipe.considerations = (pipe.considerations,)
            
    # Attempt to use h_ext if it's defined
    if getattr(pipe, 'h_ext', None) is not None:
        try:
            h = pipe.h_ext.m_as(ureg.W / ureg.K / ureg.m ** 2) * ureg.W / ureg.K / ureg.m ** 2
        except AttributeError:
            raise Exception('h_ext is defined but does not have the expected type or methods.')
    else:
        # Use considerations if h_ext is not defined
        considerations = considerations or getattr(pipe, 'considerations', None)
        
        if considerations is None:
            raise Exception('You should provide considerations to calculate the external heat transfer coefficient.')
        
        # Convert considerations to a set for faster lookup
        considerations = set(considerations)
        
        # Initialize heat transfer coefficient
        h = 0

        if 'convection' in considerations: #Convection considered
            
            #Calculate Rayleigh and Prandtl numbers
            Ra_ = Ra(fluid, T_wall, pipe.OD)
            Pr_ = Pr(fluid)

            #Determine orientation of the pipe      
            try:
                orientation = pipe.orientation
            except:
                orientation = None
            #Calculatue Nusselt numbers  
            if orientation == 'vertical':        
                Nu_ = Nu_vcyl(Pr_, Ra_, pipe.OD, pipe.L)
            elif orientation == 'horizontal':
                Nu_ = Nu_hcyl(Pr_, Ra_)
            else:
                #Use maximum Nusselt if orientation is not defined
                Pr_ = max(Pr_, 0.7)  # Set a lower bound for Prandtl number ##Rosalyn- added Prandlt minimum value to fit within interpolation range
                Nu_ = max (Nu_vcyl(Pr_, Ra_, pipe.OD, pipe.L), Nu_hcyl(Pr_, Ra_))
            
            #Calculate heat transfer coefficient of external fluid
            h_conv = heat_trans_coef(fluid, Nu_, pipe.OD)
            print(' ')
            print('conv')
            print(h_conv)
            h = h_conv

        if 'radiation' in considerations:  #radiation considered
            
            #Determine emissivity of material
            try:
                epsilon = pipe.epsilon
            except:
                #Assume material is polished steel if not defined
                epsilon = 0.075 
            
            #Calculate radiation heat transfer coefficient
            sigma = 5.670373e-8 * ureg.W/ureg.m ** 2 / ureg.K ** 4
            h_rad = epsilon * sigma * (fluid.T ** 4 - T_wall ** 4) / (fluid.T - T_wall) 
            
            #Calculate total heat transfer coefficient 
            print(' ')
            print('rad')
            print(h_rad)
            h = h + h_rad

        if 'icing' in considerations: ###including Rad ice and h_ice specific in the problem: to do 
            print(' ')    
            print(T_wall)
            
            # Define Reference External Fluid at Film Temp
            T_film = (fluid.T + T_wall )/ 2 
            fluid_film = fluid.copy() 
            fluid_film.update('P', fluid.P ,'T', T_film)
            print(T_film)
            
            # Mass Transport Considerations
            mass_transport = 0.225*((T_film.m_as(ureg.K)/273.15)**1.8)/10000*ureg.m**2/ureg.s ### Equation from Erik's slides and found in references ___ #Rosalyn enter references

            #Calculate Fluid Properties for Film Temp Fluid
            vis = 0.00001716 *((T_film.m_as(ureg.K)/273.15)**(3/2))*((273.15+110.4)/(T_film.m_as(ureg.K)+110.4)) *(ureg.kg/ureg.s/ureg.m) ### Empirical Formula for viscosity- works only for air
            Sc_f = vis /(mass_transport * fluid_film.Dmass) #calculated differently than excel sheet: density is slightly different
            Pr_f = Pr(fluid_film) #calculated differently than excel sheet

            # Lewis number and relationship combining mass and heat transfer properties
            Le_ = Sc_f/Pr_f 
            lewis_rel = Le_**(-2/3)/(fluid_film.Dmass*fluid_film.Cpmass) 

            # Concentration calculations: surrounding air must have higher concentration than at the pipe surface for condensation to occur
            # use reference function fluid_interp to interpolate from enthalpy and density values from REFPROP
            surface_concen = interp_density(T_wall)
     
            # use reference water fluid to calculate concentration and enthalpy of water vapor in air
            fluid_vapor = ThermState('water', P= fluid.P , T=fluid.T)
            air_vapor_concen = (2340*ureg.Pa *humidity*fluid_vapor.molar_mass)/(fluid_vapor.gas_constant*fluid_vapor.T) #ideal gas law- sublimation pressure calculated in ctb? 
            concentration_grad = air_vapor_concen - surface_concen
            
            # enthalpy calculations: interpolate from REFPROP values
            surface_enthalpy = interp_enthalpy(T_wall)
            vapor_enthalpy = interp_enthalpy(fluid.T)
            
            # add latent heat of sublimation to delta_H if going from ice to vapor
            if T_wall < 294 *ureg.K:
                latent_heat = 2830 *ureg.kJ/ureg.kg
                delta_H =  vapor_enthalpy - surface_enthalpy + latent_heat
            else:
                delta_H = vapor_enthalpy - surface_enthalpy
            
            # calculate the mass flux of condensation
            mass_flux = concentration_grad * lewis_rel * h_conv 
    
            # Standard heat transfer equations to calculate heat flux and coefficient - add assumptions/derivations?
            q_icing = mass_flux*delta_H
            h_icing = q_icing/(fluid.T-T_wall)
            
            # update heat transfer coeff value
            h = h + h_icing
            
        if h == 0:
            raise ValueError("The calculated heat transfer coefficient is zero, indicating invalid or missing considerations.")

    return h.to(ureg.W / ureg.K / ureg.m ** 2)


def interp_density(temp):
    interp =  interp1d(rp_temp, rp_density, kind = 'linear' , fill_value = "extrapolate")     
    density_value = interp(temp)*ureg.kg/(ureg.m**3)
    return density_value
    
def interp_enthalpy(temp): 
    interp =  interp1d(rp_temp, rp_enthalpy, kind = 'linear' , fill_value = "extrapolate")     
    enthalpy_value = interp(temp)*ureg.kJ/ureg.kg     
    return enthalpy_value
     

def test_function_icing():
    #### Parameters
    OD = Q_('8 inch')
    pipe = ht.piping.Pipe(OD, SCH=10, L=5*ureg.m, eps=0.000045*ureg.m)  # NPS pipe class
    # m_dot = Q_('50 g/s')  # Mass flow
    fluid = ht.ThermState('helium', T=100*ureg.K, P=1*ureg.bar)  # Fluid state
    pipe.epsilon = 0.96  # for steel < 273 K
    considerations = ('convection', 'radiation', 'icing')
    
    # Create a range of wall temperatures, similar to T_ext in the previous example
    T_wall_values = np.linspace(93, 293, 200) * ureg.K  # Range from 0 to 293 K
    heat_transfer_coefficients = []  # List to store the calculated heat transfer coefficients

    # Loop over each T_wall value
    for T_wall in T_wall_values:
        # Call a function to calculate heat transfer coefficient
        h = h_ext_(fluid, pipe, T_wall, considerations)  
        
        # Store the result
        heat_transfer_coefficients.append(h)

    # Return both the temperature values and their corresponding heat transfer coefficients
    return T_wall_values, heat_transfer_coefficients

















