#==============================================================
# Calculates the integration errors on y' = exp(y-2x)+2
#
#==============================================================
# required libraries
import numpy as np               # numerical routines (arrays, math functions etc)
import matplotlib.pyplot as plt  # plotting commands
import utilities as util     # for rescaleplot

import ode_integrators as odeint # contains the drivers for ODE integration.
import ode_step as step          # contains the ODE single step functions.

#==============================================================
# function dydx = exponential(x,y,dx)
#
# Calculates RHS for error test function
#
# input: 
#   x,y    : 
#
# global:
#   -
# output:
#   dydx    : vector of results as in y'=f(x,y)
#--------------------------------------------------------------

def get_dydx(x, y, dx):
    dydx = np.exp(y[0] - 2.0 * x) + 2.0
    return dydx

#==============================================================
# main
#==============================================================
def main():

#   the function should accomplish the following:
#   Test the 3 fixed stepsize integrators euler,rk2,rk4 by calculating their 
#   cumulative integration errors for increasing step sizes
#   on the function y' = exp(y(x)-2*x)+2. This is given in the function "exponential" above.
#   Use the integration interval x=[0,1], and the initial 
#   condition y[0] = -ln(2).
#   (1) define an array containing the number of steps you want to test. Logarithmic spacing
#       (in decades) might be useful.
#   (2) loop through the integrators and the step numbers, calculate the 
#       integral and store the error. You'll need the analytical solution at x=1,       
#       see homework assignment.
#   (3) Plot the errors against the stepsize as log-log plot, and print out the slopes.

    fINT  = odeint.ode_ivp                  # use the initial-value problem driver
    fORD  = [step.euler,step.rk2,step.rk4]  # list of stepper functions to be run
    fRHS  = get_dydx                        # the RHS (derivative) for our test ODE
    fBVP  = 0                               # unused for this problem

    #??????????????????????????????????????????????????????????
    
    x_start = 0
    x_stop = 1
    y0 = -np.log(2)
    y1_true = 2.0
    nstep = np.array([10, 100, 1000, 10000, 100000])
    
    def f(x):
        return 2*x - np.log(2 - x)
    
    n = nstep.size
    error_list = []

    for i in range(n):    
        for j in range(3):
            x, y, it = fINT(fRHS, fORD[j], fBVP, x_start, y0, x_stop, nstep[i])
            error_list.append(y[0, nstep[i]] - y1_true)
    error = np.array(error_list) 
     
    index = np.arange(n)
    euler_index = 3 * np.arange(n)
    rk2_index = 3 * np.arange(n) + 1
    rk4_index = 3 * np.arange(n) + 2
    
    euler_error = np.zeros(n)
    rk2_error = np.zeros(n)
    rk4_error = np.zeros(n)
    
    euler_error[index] = error[euler_index]
    rk2_error[index] = error[rk2_index]
    rk4_error[index] = error[rk4_index]

    x = nstep
    plt.xlabel('log(step number)')
    plt.ylabel('log(error)')
    plt.loglog(x, np.abs(euler_error), '-', x, np.abs(rk2_error), '-', x, np.abs(rk4_error), '-')
    plt.legend(('Euler Error', 'rk2 Error', 'rk4 Error'),loc = 0)
    plt.grid(True)
    plt.show()
    
    # SLOPES
    
    euler_slope = (np.log(np.abs(euler_error[4])) - np.log(np.abs(euler_error[0]))) / (100000 - 10)
    rk2_slope = (np.log(np.abs(rk2_error[4])) - np.log(np.abs(rk2_error[0]))) / (100000 - 10)
    rk4_slope = (np.log(np.abs(rk4_error[4])) - np.log(np.abs(rk4_error[0]))) / (100000 - 10)
    
    slopes = 'Logarithmic Slope of the Euler Slope: ' + str(euler_slope)
    slopes += '\n'
    slopes += 'Logarithmic Slope of the rk2 Slope: ' + str(rk2_slope)
    slopes += '\n'
    slopes += 'Logarithmic Slope of the rk4 Slope: ' + str(rk4_slope)
    print(slopes)
        
    #??????????????????????????????????????????????????????????

#==============================================================

main()