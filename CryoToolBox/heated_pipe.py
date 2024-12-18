# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:51:02 2024

@author: rbeckwit
"""


"""Pressure drop and heat transfer calculation.
"""

from math import pi, sin, log, log10, sqrt, tan, tanh     
from .std_conditions import ureg, Q_, P_NTP
from .functions import Re, Ra, Nu_vcyl, Nu_hcyl, Ra_update    
from .functions import Material, Property     
from .functions import nist_property, conduction_cyl     
from scipy.optimize import root_scalar, minimize  
from .functions import AIR
from .functions import heat_trans_coef, Ra, Nu_vcyl, Nu_hcyl, Pr
from .cp_wrapper import ThermState
from .piping import Mach, Mach_total, K_lim, ChokedFlow, HydraulicError, velocity, dP_Darcy, dP_adiab, Pipe, Tube, CopperTube, dP_incomp
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd


from . import os, __location__

water_ice_table_name = 'Water_ice_data.txt'


# Import data for icing calculations: located in text file refprop_data.txt
refprop_data = pd.read_csv(os.path.join(__location__, water_ice_table_name), sep ='\s+', engine = 'python')
rp_temp = refprop_data['Temperature']
rp_density = refprop_data['Density']
rp_enthalpy = refprop_data['Enthalpy']

class pipe_properties:
    """
    Class to define the necessary isolation inputs.
    """
    def __init__(self, k, OD, T_ext=293 * ureg.K):
        self.k = k
        self.OD = OD
        self.T_ext = T_ext     

### Seperate friction factor and Nusselt calculations:
def Nusselt_laminar(Re_, Pr_, L_ID):
    """
    Non dimentional calculation of the Nusselt in pipe in laminar flow  
    Bennet, T.D., Journal of Heat Transfer, 2020
    """
    # Verify input conditions     
    if Pr_ < 0.1:
        raise ValueError(f'Prandtl number (Pr) must be > 0.1. The value is {Pr_}')  

    # Calculate and verify Graetz number and Inverse Graetz number
    SGZ = L_ID / Re_
    GZ = L_ID / (Re_ * Pr_)
    if GZ < 1e-6:
        raise ValueError(f'Inverse Graetz number (GZ) must be > 1e-6. The value is {GZ}')
        
    # Calculate Nusselt numbers: temperature constant and flux constant conditions
    Nu_T = ((5.001 / GZ**1.119 + 136.0)**0.2978 - 0.6628) / tanh(2.444 * SGZ**(1 / 6) * (1 + 0.565 * SGZ**(1 / 3)))
    Nu_Q = ((6.562 / GZ**1.137 + 220.4)**0.2932 - 0.5003) / tanh(2.530 * SGZ**(1 / 6) * (1 + 0.639 * SGZ**(1 / 3)))
    
    return Nu_T, Nu_Q

def Nusselt_turbulent(f, Re_, Pr_, L_ID):
    """
    Non dimentional calculations of the Nusselt in pipe in turbulent flow following Section 5.2.3 of Nellis and Klein (2020)
    """
    # Verify input conditions
    if Pr_ < 0.004 or Pr_ > 2000:
        raise ValueError(f'Prandtl number (Pr) must be between 0.004 and 2000. The value is {Pr_}')  
    
    # Specific friction number
    friction = f / (1 + (1 / L_ID)**0.7)
    
    # Nusselt, Gnielinski, Int. Chem. Eng., 1976
    Nusselt = ((friction / 8) * (Re_ - 1000) * Pr_) / (1 + 12.7 * sqrt(friction / 8) * (Pr_**(2 / 3) - 1))
    
    # Correct Nusselt number for low Prandtl numbers, Notter & Sleicher, Chem. Eng. Sci., 1972
    if Pr_ < 0.5:
        Nusselt_low_pr = 4.8 + 0.0156 * Re_ ** 0.85 * Pr_ ** 0.93
        if Pr_ < 0.1:
            Nusselt = Nusselt_low_pr
        else:
            Nusselt = Nusselt_low_pr + (Pr_ - 0.1) * (Nusselt - Nusselt_low_pr) / 0.4  
            
    # Correct Nusselt for developing flow
    Nu = Nusselt * (1 + (1 / L_ID)**0.7)
    
    return Nu

def Nusselt(f, Re_, Pr_, L_ID):
    """
    Non dimentional calculations of the Nusselt in pipe following Section 5.2 of Nellis and Klein (2020)
    General function that uses `Nusselt_turbulent` and `Nusselt_laminar` to calculate the Nusselt number of the system. 
    """
    # Check Flow conditions
    if Re_ < 0.001 or Re_ > 5e7: 
        raise ValueError(f'Reynolds number (Re) must be between 0.001 and 5E7. The value is {Re_}')

    if Re_ > 3000:  # Turbulent flow
        Nu_T = Nusselt_turbulent(f, Re_, Pr_, L_ID)
        Nu_Q = Nu_T
        
    elif Re_ < 2300:  # Laminar flow
        Nu_T, Nu_Q = Nusselt_laminar(Re_, Pr_, L_ID)
        
    else:  # Transitional flow (Re between 2300 and 3000)
        Nu_T_turbulent = Nusselt_turbulent(f, 3000, Pr_, L_ID)
        Nu_T_laminar, Nu_Q_laminar = Nusselt_laminar(2300, Pr_, L_ID)
        
        # Interpolate between laminar and turbulent values
        interp_value = (Re_ - 2300) / (3000 - 2300) 
        Nu_T = Nu_T_laminar + interp_value * (Nu_T_turbulent - Nu_T_laminar)
        Nu_Q = Nu_Q_laminar + interp_value * (Nu_T_turbulent - Nu_Q_laminar)

    return Nu_T, Nu_Q

def heat_trans_coef_internal_surface(m_dot, fluid, pipe):
    """
    Integrates the functions `Nusselt` and `heat_trans_coef` for convenient 
    method to calculate the convective heat transfer coefficient between the
    internal flow and the inner wall of the pipe.
    
    Parameters
    ----------
    `fluid` : ThermState | Inlet fluid conditions   
    `m_dot` : Quantity {mass: 1, time: -1} | mass flow rate
    `pipe` : Pipe | defines the pipe characteristics 
    
    Returns
    -------  
    `h_T`, `h_Q`: Quantity: {mass : 1, temperature : -1, time : -3}
         | Heat transfer coefficients
     -------
    """
    # Calculate pipe and flow conditions
    Re_ = Re(fluid, m_dot, pipe.ID, pipe.area)
    L_ID = pipe.L.m_as(ureg.m)/pipe.ID.m_as(ureg.m)
    Pr_ = fluid.Prandtl

    # Calculate friction factor and Nusselt numbers
    f = (pipe.K(Re_)/L_ID)
    Nu_T, Nu_Q = Nusselt(f.to_base_units().magnitude, Re_, Pr_, L_ID)
        
    # Calculate heat transfer coefficients: temperature and heat flux constant conditions
    h_T = heat_trans_coef(fluid, Nu_T, pipe.ID)
    h_Q = heat_trans_coef(fluid, Nu_Q, pipe.ID)
    
    return h_T, h_Q


def wall_temperature(x, T_avg, pipe, h_coeff, m_dot, condition): 
  
    """
    Description
    ---------- 
    This function is used to find the wall temperature of a pipe. 
    The system can either have a defined applied:
    - Heat flux (`pipe_Q_def`), 
    - Pipe outer wall temperature (`pipe_Tw_def`), 
    - External fluid heat transfer coefficient (`pipe_h_ext`),
    - Pipe insulation (`pipe_insulated`).  
    <div style="text-align: center; margin: 0; padding: 0;">
        <figure style="text-align: center; margin: 0; padding: 0;">
            <img src="find_Tw.jpg" alt="find_Tw" style="max-width: 100%; height: auto; margin: 0 auto; display: block;" />
            <figcaption style="text-align: center; font-style: italic; margin-top: 10px;">Figure 1: </figcaption>
            <br>
        </figure>
    </div>   
    Depending on the inputs, either the inner or outer wall temperature is calculated. The heat flux, $q"$ is calculated 
    from the general convection equation as a function of the unknown wall temperature shown below in Equation #. Here $h$ is the heat transfer
    coefficient, $T_{avg}$ is the average fluid temperature, and $Tw_i$ and $Tw_o$ are the inner and outer wall temperatures respectively.      
    $$ q"_{conv} = h * (Tw_i – T_{avg})  \qquad   {(1)} $$ 
    Similarly, the heat flux is also calculated as a function of the unknown wall temperature from the conduction equation.  
    $$ q"_{cond} = (Tw_o – Tw_i)/ R_L \qquad {(2)} $$ 
    $ R_L $ is the thermal resistance of the wall defined below in Equation # where $k$ is the thermal conductivity of the wall (calculated using the integral method), $L$ is the length of the pipe, 
    and $D_o$ and $D_i$ are the outer and inner diameters of the pipe respectively. 
    $$ R_L = log(D_o/D_i) / (2 * pi * L * k) \qquad {(3)} $$ 
    These two expressions for heat flux are set equal to each other and squared to create a quadratic.     
    $$ (q"_{conv} - q"_{cond})^2 = 0 \qquad {(4)} $$ 
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
        h_ext = heat_trans_coef_external_surface(fluid_external, pipe, x * ureg.K)     
        
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
    
    k = conductivity(pipe, Tw_o, Tw_i)
    Q_cond = conduction_cyl(pipe.ID.to(ureg.m), pipe.OD.to(ureg.m), pipe.L.to(ureg.m), k, (Tw_o - Tw_i))
    Q_conv = -h_coeff * (T_avg - Tw_i) * pipe.ID.to(ureg.m) * pipe.L.to(ureg.m) * 3.14  
    
    return (Q_conv - Q_cond).m_as(ureg.W) ** 2  


def wall_temps_constant_heat_flux(fluid, pipe, m_dot, dP, h_Q):
    """
    Description
    ----------    
    This function is used for systems with a defined constant heat flux. The set-up of the function is displayed below in Figure #. 
    <div style="text-align: center; margin: 0; padding: 0;">
        <figure style="text-align: center; margin: 0; padding: 0;">        
            <img src="pipe_Q_def.jpg" alt="pipe_Q_def" style="max-width: 100%; height: auto; margin: 0 auto; display: block;" />
            <figcaption style="text-align: center; font-style: italic; margin-top: 10px;">Figure 2: </figcaption>
            <br>
        </figure>
    </div>       
    The heat flux is used to calculate the fluid’s change of enthalpy. This is displayed in Equation # shown below where $dH$ is the 
    change in enthalpy, $q"$ is the defined heat flux, $S_o$ is the pipe's outer surface area, and $m_{dot}$ is the mass flow rate.  
    $$ dH = q" * S_o/m_{dot} \qquad {(5)} $$
    Using the enthalpy change and the input pressure drop value, the fluid conditions at the outlet are calculated. 
    The average temperature of the fluid is determined from the inlet and outlet temperatures and assumes a linear temperature gradient.   
    This average temperature is used to calculate the inner wall temperature using the general convection equation shown below in Equation #. 
    $$ Tw_i = q"/(h_Q * S_i / S_o) + T_{avg}  \qquad {(6)} $$
    In the above expression, $S_i$ is the pipe's inner surface area, $h_Q$ is the heat transfer coefficient calculated using function `heat_trans_coef_internal_surface` (constant heat flux), 
    and $T_{avg}$ is the avaerage temperatue of the fluid. Finally, the outer wall temperature is calculated using function `wall_temperature`.
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
    
    #Calculate Tw_i and Tw_o: minimum of the quadratic find_Twall
    Tw_i = T_avg + pipe.Q_def / h_Q * (pipe.OD / pipe.ID)
    if pipe.Q_def == 0 * ureg.W / ureg.m ** 2:
        Tw_o = Tw_i
    else:
        # Assign bounds for minimize function to operate within
        dT = log(pipe.OD/pipe.ID)*(pipe.Q_def*pipe.OD)/(2*conductivity(pipe,Tw_i))
        
        # Calculate Tw_o: minimum of the quadratic find_Twall
        condition = 1 # condition to designate Q_def in find_Twall
        Tw_o = root_scalar(wall_temperature, x0 = (Tw_i + dT/2).m_as(ureg.K), x1 = (Tw_i + dT).m_as(ureg.K), args=(T_avg, pipe, h_Q, m_dot, condition)).root * ureg.K

    return Tw_i, Tw_o  
 

# Define main functions
def heated_properties_constant_wall_temp(fluid, pipe, m_dot, dP, h_T):
    """
    Description
    ----------    
    This function is used for systems with a constant pipe outer wall temperature. 
    The set-up is shown below in Figure # with the inputs highlighted.  
    <div style="text-align: center; margin: 0; padding: 0;">
        <figure style="text-align: center; margin: 0; padding: 0;">
            <img src="pipe_Tw_def.jpg" alt="pipe_Tw_def" style="max-width: 100%; height: auto; margin: 0 auto; display: block;" />
            <figcaption style="text-align: center; font-style: italic; margin-top: 10px;">Figure 3: </figcaption>
            <br>
        </figure>
    </div>           
    The inner wall temperature is calculated in `find_Twall` from the average temperature which is initially assigned to a reference value.
    The inner wall temperature is used in Equation # described in `pipe_Q_def` to calculate the change in enthalpy of the fluid.
    The outlet pressure is calculated in `dP_Pipe`. Additionally, the outlet pressure is calculated in `dP_pipe`. 
    Knowing these fluid conditions, the outlet temperature of the fluid is determined.
    The average temperature is then re-calculated from the inlet and outlet fluid temperatures using Equation #.  
    $$ T_{avg} = (T_{inlet} + T_{outlet})/2 \qquad {(7)} $$
    This value replaces the reference value, and the code loops until the reference temperature and the calculated average temperature converge.
    Once $T_{avg}$ is known, the heat flux is calculated from the general convection equation shown below in Equation # where $h_T$ is the heat transfer coefficient calculated in `dP_Pipe`. 
    $$q” = -h_T * (T_{avg}-Tw_i) \qquad {(8)} $$ 
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
            
        # Calculate Tw_i:  minimum of the quadratic find_Twall
        condition = 2 # condition to designate Tw_def in find_Twall

        Tw_i = minimize(wall_temperature,  x0 = T_avg.m_as(ureg.K), args = (T_avg, pipe, h_T, m_dot, condition), bounds=[bracket]).x[0] *ureg.K        
        
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


def heated_properties_constant_surface_heat_trans_coef(fluid, pipe, m_dot, dP, h_T): 
    """
    Description
    ----------    
    This function is used for systems with defined external conditions. Either the heat transfer coefficient of the external fluid can be defined, or 
    methods of heat transfer to be considered can be specified. This is computed and explained in `heated_pipe`.    
    The set-up for this function is displayed below in Figure # with the input variables highlighted. 
    <div style="text-align: center; margin: 0; padding: 0;">
        <figure style="text-align: center; margin: 0; padding: 0;">
            <img src="pipe_h_ext.jpg" alt="pipe_h_ext" style="max-width: 100%; height: auto; margin: 0 auto; display: block;" />
            <figcaption style="text-align: center; font-style: italic; margin-top: 10px;">Figure 4: </figcaption>
            <br>
        </figure>
    </div>    
    The outer wall temperature, $Tw_o$, is calculated in `find_Twall` from the average temperature, $T_{avg}$, which is initially assigned to a reference value.
    The expression for the heat flux due to convection of the external fluid and the outer wall is defined below in Equation # where $h_{ext}$ is the external 
    heat transfer coefficient, $D_o$ is the outer diameter, $D_i$ is the inner diameter, and $T_{ext}$ is the external fluid temperature. 
    The external fluid heat transfer coefficient is determined in the function `h_ext_`.
    $$ q" = (T_{ext}-Tw_o) * h_{ext} * S_o \qquad {(9)} $$
    The heat flux can also be calculated from the convection between the internal fluid and the inner wall defined below in Equation #. 
    $$ q" = (Tw_i - T_{avg}) * h_T * S_i \qquad {(10)} $$
    Combining and simplifying these equations, an expression for the inner wall temperature, Equation #, is found.  
    $$Tw_i = [h_{ext} * D_o * (T_{ext} - Tw_o) / (h_T * D_i)] + T_{avg} \qquad {(11)} $$
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
        Tw_o = minimize(wall_temperature, x0=(fluid_external.T + T_avg).m_as(ureg.K)/ 2, args = (T_avg, pipe, h_T, m_dot, condition), bounds=bracket).x[0] * ureg.K                                                  
        
       
        # Caclulate external heat transfer coefficient for system: calculates h_ext for system with h_type defined otherwise uses defined h_ext
        h_ext = heat_trans_coef_external_surface(fluid_external, pipe, Tw_o)
        
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
    
        return Tw_i, Tw_o, Q #, h_ext #Here

def heated_properties_insulated(fluid, pipe, m_dot, dP, h_T): 
    """
    Description
    ----------    
    This function is used for piping systems with insulation. The insulation must have a defined thermal conductivity and outer diameter. 
    The external temperature of the insulation is assumed to be $293K$ if not specifically defined.   
    The set-up of the function is shown below in Figure # where the variable inputs to the function are highlighted.
    <div style="text-align: center; margin: 0; padding: 0;">
        <figure style="text-align: center; margin: 0; padding: 0;">
            <img src="pipe_insulated.jpg" alt="pipe_insulated" style="max-width: 100%; height: auto; margin: 0 auto; display: block;" />
            <figcaption style="text-align: center; font-style: italic; margin-top: 10px;">Figure 5: </figcaption>
            <br>
        </figure>
    </div>
    The outer wall temperature, $Tw_o$, is calculated in `find_Twall` from the average temperature, $T_{avg}$ , which is initially assigned to a reference value.
    The expression for the heat flux due to convection of the internal fluid and the inner wall is defined below in Equation # where $h_{ext}$ is the external 
    heat transfer coefficient, $D_o$ is the outer diameter, $ D_i $ is the inner diameter, and $T_{ext}$ is the external fluid temperature.
    $$ q" = (Tw_i - T_{avg}) * h_T * S_i \qquad {(12)} $$
    The heat flux can also be calculated from the conduction between the insulation with the outer wall of the pipe defined below in Equation # where $R_L$ is the thermal resistance of the wall. 
    $$ q" = (T_{ext}-Tw_o) / R_L \qquad {(13)} $$
    Combining and similifying these equations, an expression for the inner wall temperature, Equation #, is found.  
    $$ Tw_i = ((k_{ins}* (T_{ext} - Tw_o) )/ (h_T * D_i * log(D_o/D_i)) +T_{avg} \qquad {(14)} $$ 
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
         
        ###Calculate Tw_i and Tw_o: minimum of the quadratic find_Twall
        condition = 4 # communicate the heat transfer condition: defined_insulation
        Tw_o = minimize(wall_temperature, x0=pipe.isolation.T_ext.m_as(ureg.K) - (pipe.isolation.T_ext.m_as(ureg.K) - T_avg.m_as(ureg.K)), args = (T_avg, pipe, h_T, m_dot, condition), bounds=bracket).x[0] * ureg.K 
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


def heated_properties(pipe, fluid, m_dot):
    """Determine the heated status of the piping component
    """
    ### Calculate pressure drop and heat transfer coefficient
    #dP, h_T, h_Q = dP_Pipe(m_dot, fluid, pipe)  
    dP = dP_incomp(m_dot, fluid, pipe) ###to modify later
    h_T, h_Q = heat_trans_coef_internal_surface(m_dot, fluid, pipe)
    
    ###heat flux defined on the wall of the pipe
    if hasattr(pipe, 'Q_def') and pipe.Q_def != None :
        try: 
            pipe.Q_def.m_as(ureg.W / ureg.m ** 2)
        except:
            raise ValueError(f"the Q_def is not properly defined in component {pipe}" )
        Tw_i, Tw_o = wall_temps_constant_heat_flux(fluid, pipe, m_dot, dP, h_Q)
        Q = pipe.Q_def
        
    ###Temperature defined on the wall of the pipe
    elif hasattr(pipe, 'Tw_def') and pipe.Tw_def != None :
        try: 
            pipe.Tw_def.m_as(ureg.K)
        except:
            raise ValueError(f"the Tw_def is not properly defined in component {pipe}" )
        Tw_i, Tw_o, Q = heated_properties_constant_wall_temp(fluid, pipe, m_dot, dP, h_Q)

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
        
        Tw_i, Tw_o, Q = heated_properties_constant_surface_heat_trans_coef(fluid, pipe, m_dot, dP, h_T)  #Here- add h_ext 

    ###Isolation on the external of the pipe 
    elif hasattr(pipe, 'isolation') and pipe.isolation != None :
        try: 
            pipe.isolation.k.m_as(ureg.W / ureg.m / ureg.K)
            pipe.isolation.OD.m_as(ureg.m)
        except:
            raise ValueError(f"the isolation (k, OD) is not properly defined in component {pipe}" )
        Tw_i, Tw_o, Q = heated_properties_insulated(fluid, pipe, m_dot, dP, h_T)
                
    ###Other
    else: 
        pipe.Q_def = Q = 0 * ureg.W/ ureg.m ** 2
        Tw_i, Tw_o = wall_temps_constant_heat_flux(fluid, pipe, m_dot, dP, h_Q)
    
    return Tw_i.to(ureg.K), Tw_o.to(ureg.K), dP.to(ureg.bar), Q.to(ureg.W/ ureg.m ** 2) #, h_ext.to(ureg.W/ureg.m**2/ureg.K) # Here
        

def conductivity(pipe, T_wall, T_ext=293 * ureg.K):     ### you should add the possibility to define a table of values to do that (for materials not in the database)
    """Determine the thermal conductivity of the component   
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


def heat_trans_coef_external_surface(fluid, pipe, T_wall, considerations=None, humidity=0.5): ### you should add the possibility to define a table of values to do that
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

        if 'convection' in considerations: # sccounting for convection
            
            # Define film temperature to calculate fluid properties
            T_film = (fluid.T + T_wall )/ 2 
            fluid_film = fluid.copy() 
            fluid_film.update('P', fluid.P ,'T', T_film)
            
            # calculate flow conditions
            Pr_ = Pr(fluid_film)
            Ra_ = Ra_update(fluid, T_wall, pipe.OD)

            #Determine orientation of the pipe and calculate Nusselt number   
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
                Nu_ = max (Nu_vcyl(Pr_, Ra_, pipe.OD, pipe.L), Nu_hcyl(Pr_, Ra_))
            
            #Calculate heat transfer coefficient at the external surface of the pipe
            h_conv = heat_trans_coef(fluid_film, Nu_, pipe.OD)
            
            # For determining total heat transfer coefficient
            h = h_conv

        if 'radiation' in considerations:  # accounting for radiation
            
          
            # #Assume material is stainless steel
            if T_wall < 274*ureg.K: 
                #Determine emissivity of material
                try:
                    epsilon = pipe.epsilon
                except:
                    epsilon = 0.35
            else: 
                # emissivity of ice if pipe is below freezing 
                epsilon = 0.96
            
            #Calculate radiation heat transfer coefficient
            sigma = 5.670373e-8 * ureg.W/ureg.m ** 2 / ureg.K ** 4
            h_rad = epsilon * sigma * (fluid.T ** 4 - T_wall ** 4) / (fluid.T - T_wall) 
            
            #Calculate total heat transfer coefficient 
            h = h + h_rad

        if 'icing' in considerations:
            """Uses the Lewis number and relationship to calculate the heat transfer and mass transfer relationship. Function neglects the ice thickness and assumes constant state in which the ice is about to condense. 
            Considers the worst case situation: neglects insulation an ice layer would produce. 
            """

            # Mass Transport Considerations
            mass_transport = 0.225*((T_film.m_as(ureg.K)/273.15)**1.8)/10000*ureg.m**2/ureg.s ### Equation from Erik's slides and found in references ___ 
            
            #Calculate Fluid Properties using the Film Temp 
            vis = 0.00001716 *((T_film.m_as(ureg.K)/273.15)**(3/2))*((273.15+110.4)/(T_film.m_as(ureg.K)+110.4)) *(ureg.kg/ureg.s/ureg.m) ### Sutherland's Formula to calculate viscosity of air
            print(vis)
            print(fluid_film.viscosity.to(ureg.kg/ureg.s/ureg.m)) #why not using this value?
            Sc_ = vis /(mass_transport * fluid_film.Dmass) # Schmidt number
            
            # Lewis number and Lewis relationship correlating mass and heat transfer properties
            Le_ = Sc_/Pr_ 
            lewis_rel = Le_**(-2/3)/(fluid_film.Dmass*fluid_film.Cpmass) 
            
            # use reference function to interpolate enthalpy and density values from REFPROP
            surface_concen = interp_density(T_wall.m_as(ureg.K))
            air_vapor_concen = interp_density(fluid.T.m_as(ureg.K))*humidity
            concentration_grad = air_vapor_concen - surface_concen
            
            
            surface_enthalpy = interp_enthalpy(T_wall.m_as(ureg.K))
            vapor_enthalpy = interp_enthalpy(fluid.T.m_as(ureg.K))
            delta_H = vapor_enthalpy - surface_enthalpy 
            
            # Standard heat transfer equations to calculate heat flux and coefficient 
            mass_flux = concentration_grad * lewis_rel * h_conv 
            q_icing = mass_flux*delta_H
            h_icing = q_icing/(fluid.T-T_wall)
            
            print(T_wall)
            print(h_conv)
            print(h_icing.to(ureg.W/ureg.K/ureg.m**2))
            
            # Calculate total heat transfer coefficient
            h = h + h_icing #icing is added to the other method, in this case it should be only valid if below 0

        if h == 0:
            raise ValueError("The calculated heat transfer coefficient is zero, indicating invalid or missing considerations.")

    return h.to(ureg.W / ureg.K / ureg.m ** 2)

def heat_flux_MLI(N,Th, Tc, layer_density = 21, P_res = 0.01*ureg.torr, emissivity = 0.043): # add units | number of layers, temp outer (hot) layer, temp inner (cold) layer
    "Calculate the heat flux of a component with MLI installed. Assuming Perf DAM with glass. Emissivity value is for aluminum."
    
    # Constants
    Cr = 7.07e-10 # radiation coefficient (function of reflector's material - aluminum)
    Cs = 7.3e-8 # solid conduction coefficient (for spacer material - glass)
    Cg = 1.46e4 # gas conduction coefficient (function of radiation has pressure between layers)
    
    # add magnitudes for temp and layer density and pressure -- verify units w/ graph
    q_MLI = Cr*emissivity*(Th.m_as(ureg.K)**4.67-Tc.m_as(ureg.K)**4.67)/N + Cs*layer_density**2.63*(Th.m_as(ureg.K)-Tc.m_as(ureg.K))*(Th.m_as(ureg.K)+Tc.m_as(ureg.K))/(2*(N+1)) + Cg*P_res.m_as(ureg.torr)*(Th.m_as(ureg.K)**0.52-Tc.m_as(ureg.K)**0.52)/N #Modified Lockheed Equation
    
    return q_MLI*ureg.W/ureg.m**2


def interp_density(temp):
    "Interpolate the density of water (>273K) or ice (<273K) given the temperature. REFPROP saturation and sublimation data."
    interp =  interp1d(rp_temp, rp_density, kind = 'linear' , fill_value = "extrapolate")     
    density_value = interp(temp)*ureg.kg/(ureg.m**3)
    return density_value
    
def interp_enthalpy(temp): 
    "Interpolate the enthalpy of water (>273K) or ice (<273K) given the temperature. REFPROP saturation and sublimation data. "
    interp =  interp1d(rp_temp, rp_enthalpy, kind = 'linear' , fill_value = "extrapolate")     
    enthalpy_value = interp(temp)*ureg.kJ/ureg.kg     
    return enthalpy_value