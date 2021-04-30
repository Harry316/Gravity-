
#%%
#import libaries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

#Constants
θ = (51.3811*2*np.pi)/(360) #latitude of Bath university in radians
re = np.cos(θ) * 6365316 #r for Bath 
Ve = re/(24*60*60) #veloity of Earth spinning in m/s
η = 1.8347*10**(-5) #viscosity of air
p = 1.225 #density of air
CD = 0.5 #drag coefficeint
Centri = ((Ve**2) / re)#centripetal part of g constant for all
x_line = np.linspace(0,0.65,100)

#Masses and radii of balls 
# For dataset_new (3 diff balls were dropped)
m1_new = 0.0635; m2_new = 0.0434; m3_new =0.0326; # in kg
r1_new = 0.02496/2; r2_new = 0.02197/2; r3_new = 0.01997; # in meters
# For dataset_2 (3 diff balls were dropped)
m1_2 = 0.00703; m2_2 = 0.00406; m3_2 = 0.00209#mass of the ball from dataset 2 in kg
r1_2 = 0.01199/2; r2_2 = 0.00998/2; r3_2 = 0.00798/2 #radius of the ball from dataset 2 in meters
#For dataset_1 (2 diff balls were dropped)
m1_1 = 0.007031; m2_1 = 0.002986#mass of the ball from dataset1 in kg
r1_1 = 0.01198/2; r2_1 = 0.00898/2 #radius of the ball from dataset 1 in meters

# HEIGHT AT WHICH BALLS WERE DROPPED FROM
#Heights for dataset 2
h1_2 = 0.451; h2_2 = 0.718; h3_2 = 0.970; h4_2 = 1.096; h5_2 = 1.209; h6_2 = 1.473; h7_2 = 1.763 # height at which balls were dropped in meters
#Heights for dataset 1
h3_1 = 1.1175; h2_1 = 0.798; h1_1 = 0.544 # height in meters
# Heights for new dataset
h1_new = 0.4; h2_new = 0.6; h3_new = 0.8; h4_new = 1.0; h5_new = 1.2; h6_new = 1.4; h7_new = 1.6; h8_new = 1.8; h9_new = 2.0;

#Working k (constant coefficient in rag formula) for different spheres
# K for each mass in dataset new
k1_new = 0.5*p*CD*2*np.pi*r1_new**2; k2_new = 0.5*p*CD*2*np.pi*r2_new**2; k3_new = 0.5*p*CD*2*np.pi*r3_new**2
# K for dataset 1
k1_1 = 0.5*p*CD*2*np.pi*r1_1**2; k2_1 = 0.5*p*CD*2*np.pi*r2_1**2;
# K for dataset 2
k1_2 = 0.5*p*CD*2*np.pi*r1_2**2; k2_2 = 0.5*p*CD*2*np.pi*r2_2**2; k3_2 = 0.5*p*CD*2*np.pi*r3_2**2;

#imput times (for x axis) for dataset 2
#Each list contains the time (all trials) for each mass and height
times2 = pd.read_excel('Free_fall_data_MASTER.xlsx', sheet_name="data_2")
m1_h1_2 = times2["m1_h1"].dropna(); m2_h1_2 = times2["m2_h1"].dropna(); 
m1_h2_2 = times2["m1_h2"].dropna(); m2_h2_2 = times2["m2_h2"].dropna(); m3_h2_2 = times2["m3_h2"].dropna();
m1_h3_2 = times2["m1_h3"].dropna(); m2_h3_2 = times2["m2_h3"].dropna(); m3_h3_2 = times2["m3_h3"].dropna();
m1_h4_2 = times2["m1_h4"].dropna(); m2_h4_2 = times2["m2_h4"].dropna();
m1_h5_2 = times2["m1_h5"].dropna(); m2_h5_2 = times2["m2_h5"].dropna(); m3_h5_2 = times2["m3_h5"].dropna();
m1_h6_2 = times2["m1_h6"].dropna(); m2_h6_2 = times2["m2_h6"].dropna(); m3_h6_2 = times2["m3_h6"].dropna();
m1_h7_2 = times2["m1_h7"].dropna(); m2_h7_2 = times2["m2_h7"].dropna(); 
m1_yerr_2 = times2["m1_yerr"].dropna();
m2_yerr_2 = times2["m2_yerr"].dropna();
m3_yerr_2 = times2["m3_yerr"].dropna();

#imput times (for x axis) for 1
times1 = pd.read_excel('Free_fall_data_MASTER.xlsx', sheet_name="data_1")
m1_h3_1 = times1["m1_h1"].dropna(); m2_h3_1 = times1["m2_h1"].dropna(); 
m1_h2_1 = times1["m1_h2"].dropna();
m1_h1_1 = times1["m1_h3"].dropna();
m1_yerr_1 = times1["m1_yerr"].dropna();
m2_yerr_1 = times1["m2_yerr"].dropna();

#imput times (for x axis) for dataset new
# It reads the the values under that column name and sets assigns it as the time at which said mass drops at said height. 
#Essentially it's a list that contains the trials for each mass and height
timesnew = pd.read_excel('Free_fall_data_MASTER.xlsx', sheet_name="New")
m1_h1_new = timesnew["LB_40"].dropna(); m1_h2_new = timesnew["LB_60"].dropna(); m1_h3_new = timesnew["LB_80"].dropna(); m1_h4_new = timesnew["LB_100"].dropna(); m1_h5_new = timesnew["LB_120"].dropna(); m1_h6_new = timesnew["LB_140"].dropna(); m1_h7_new = timesnew["LB_160"].dropna(); m1_h8_new = timesnew["LB_180"].dropna();m1_h9_new = timesnew["LB_200"].dropna();
m2_h1_new = timesnew["MB_40"].dropna(); m2_h2_new = timesnew["MB_60"].dropna(); m2_h3_new = timesnew["MB_80"].dropna(); m2_h4_new = timesnew["MB_100"].dropna(); m2_h5_new = timesnew["MB_120"].dropna(); m2_h6_new = timesnew["MB_140"].dropna(); m2_h7_new = timesnew["MB_160"].dropna(); m2_h8_new = timesnew["MB_180"].dropna();m2_h9_new = timesnew["MB_200"].dropna();
m3_h1_new = timesnew["SB_40"].dropna(); m3_h2_new = timesnew["SB_60"].dropna(); m3_h3_new = timesnew["SB_80"].dropna(); m3_h4_new = timesnew["SB_100"].dropna(); m3_h5_new = timesnew["SB_120"].dropna(); m3_h6_new = timesnew["SB_140"].dropna(); m3_h7_new = timesnew["SB_160"].dropna(); m3_h8_new = timesnew["SB_180"].dropna();m3_h9_new = timesnew["SB_200"].dropna();


#DATA 2
#x for m1 for data 2
# The len() func returns the number of elements in the list..i.e the number of trials conducte for each mass and height.
# This is the x value data that is used when plotting 
x1_2 = [ 0,
         *([h1_2-r1_2] * len(m1_h1_2)),
         *([h2_2-r1_2] * len(m1_h2_2)),
         *([h3_2-r1_2] * len(m1_h3_2)),
         *([h4_2-r1_2] * len(m1_h4_2)),
         *([h5_2-r1_2] * len(m1_h5_2)),
         *([h6_2-r1_2] * len(m1_h6_2)),
         *([h7_2-r1_2] * len(m1_h7_2))
]

error_y1_2=[0.0005] * len(x1_2)

#t for m1
time_m1_2 = [0, *m1_h1_2, *m1_h2_2, *m1_h3_2, *m1_h4_2, *m1_h5_2, *m1_h6_2, *m1_h7_2]


# This is our expression or formula for x(t) afte consiering rag and centrifugal force
def get_g(m, r,  t, g):
    A = g + ((Ve**2)/re)
    B = 0.5*np.pi*(r**2)*CD*p/m
    return (np.log(np.cosh(t*((A*B)**0.5)))/B)

# Our formula for x(t) for mass 1 in dataset 2
def get_g1_2(t, g):
    return get_g(m1_2, r1_2, t, g)

popt1_2, pcov1_2 = curve_fit(get_g1_2, time_m1_2, x1_2, sigma = error_y1_2)
print("g for m1 for data 2 =", popt1_2[0])
print("uncertainty g for m1 for data 2 = ", pcov1_2[0]**0.5)


#x for m2 for data2
x2_2 = [ 0,
         *([h1_2-r2_2] * len(m2_h1_2)),
         *([h2_2-r2_2] * len(m2_h2_2)),
         *([h3_2-r2_2] * len(m2_h3_2)),
         *([h4_2-r2_2] * len(m2_h4_2)),
         *([h5_2-r2_2] * len(m2_h5_2)),
         *([h6_2-r2_2] * len(m2_h6_2)),
         *([h7_2-r2_2] * len(m2_h7_2))
]

error_y2_2=[0.0005] * len(x2_2)

#t for m2
time_m2_2 = [0, *m2_h1_2, *m2_h2_2, *m2_h3_2, *m2_h4_2, *m2_h5_2, *m2_h6_2, *m2_h7_2]

def get_g2_2(t, g):
    return get_g(m2_2, r2_2, t, g)

popt2_2, pcov2_2 = curve_fit(get_g2_2, time_m2_2, x2_2, sigma=error_y2_2)
print("g for m2 in data 2 =", popt2_2[0])
print("uncertainty g for m2 in data 2 = ", pcov2_2[0]**0.5)


#x for m3
x3_2 = [ 0,
         *([h2_2-r3_2] * len(m3_h2_2)),
         *([h3_2-r3_2] * len(m3_h3_2)),
         *([h5_2-r3_2] * len(m3_h5_2)),
         *([h6_2-r3_2] * len(m3_h6_2)),
]

error_y3_2=[0.0005] * len(x3_2)

#t for m3
time_m3_2 = [0, *m3_h2_2, *m3_h3_2, *m3_h5_2, *m3_h6_2]

def get_g3_2(t, g):
    return get_g(m3_2, r3_2, t, g)

popt3_2, pcov3_2 = curve_fit(get_g3_2, time_m3_2, x3_2, sigma = error_y3_2)
print("g for m3 for data 2 =", popt3_2[0])
print("uncertainty g for m3 for data 2 = ", pcov3_2[0]**0.5)


#DATA 1
#x for m1 for data 1
x1_1 = [ 0,
         *([h1_1-r1_1] * len(m1_h1_1)),
         *([h2_1-r1_1] * len(m1_h2_1)),
         *([h3_1-r1_1] * len(m1_h3_1))
]

error_y1_1=[0.005] * len(x1_1)

#t for m1
time_m1_1 = [0, *m1_h1_1, *m1_h2_1, *m1_h3_1]


def get_g1_1(t, g):
    return get_g(m1_1, r1_1, t, g)

popt1_1, pcov1_1 = curve_fit(get_g1_1, time_m1_1, x1_1, sigma = error_y1_1)
print("g for m1 for data 1 =", popt1_1[0])
print("uncertainty g for m1 for data 1 = ", pcov1_1[0]**0.5)

#x for m2 for data 1
x2_1 = [ 0,
         *([h3_1-r2_1] * len(m2_h3_1))
]

error_y2_1=[0.005] * len(x2_1)

#t for m1
time_m2_1 = [0, *m2_h3_1]


def get_g2_1(t, g):
    return get_g(m2_1, r2_1, t, g)

popt2_1, pcov2_1 = curve_fit(get_g2_1, time_m2_1, x2_1, sigma=error_y2_1)
print("g for m2 for data 1 =", popt2_1[0])
print("uncertainty g for m2 for data 1 = ", pcov2_1[0]**0.5)

#NEW DATA
#x for m1 for data NEW
x1_new = [ 0,
         *([h1_new-r1_new] * len(m1_h1_new)),
         *([h2_new-r1_new] * len(m1_h2_new)),
         *([h3_new-r1_new] * len(m1_h3_new)),
         *([h4_new-r1_new] * len(m1_h4_new)),
         *([h5_new-r1_new] * len(m1_h5_new)),
         *([h6_new-r1_new] * len(m1_h6_new)),
         *([h7_new-r1_new] * len(m1_h7_new)),
         *([h8_new-r1_new] * len(m1_h8_new)),
         *([h9_new-r1_new] * len(m1_h9_new))
]

error_y1_new=[0.004] * len(x1_new)

#t for m1
time_m1_new = [0, *m1_h1_new, *m1_h2_new, *m1_h3_new, *m1_h4_new, *m1_h5_new, *m1_h6_new, *m1_h7_new, *m1_h8_new, *m1_h9_new]


def get_g1_new(t, g):
    return get_g(m1_new, r1_new, t, g)

popt1_new, pcov1_new = curve_fit(get_g1_new, time_m1_new, x1_new, sigma=error_y1_new)
print("g for m1 for dat new =", popt1_new[0])
print("uncertainty g for m1 for data new = ", pcov1_new[0]**0.5)

#x for m2 for data NEW

x2_new = [ 0,
         *([h1_new-r2_new] * len(m2_h1_new)),
         *([h2_new-r2_new] * len(m2_h2_new)),
         *([h3_new-r2_new] * len(m2_h3_new)),
         *([h4_new-r2_new] * len(m2_h4_new)),
         *([h5_new-r2_new] * len(m2_h5_new)),
         *([h6_new-r2_new] * len(m2_h6_new)),
         *([h7_new-r2_new] * len(m2_h7_new)),
         *([h8_new-r2_new] * len(m2_h8_new)),
         *([h9_new-r2_new] * len(m2_h9_new))
]

error_y2_new=[0.004] * len(x2_new)

#t for m2
time_m2_new = [0, *m2_h1_new, *m2_h2_new, *m2_h3_new, *m2_h4_new, *m2_h5_new, *m2_h6_new, *m2_h7_new, *m2_h8_new, *m2_h9_new]


def get_g2_new(t, g):
    return get_g(m2_new, r2_new, t, g)

popt2_new, pcov2_new = curve_fit(get_g2_new, time_m2_new, x2_new, sigma=error_y2_new)
print("g for m1 for dat new =", popt2_new[0])
print("uncertainty g for m1 for data new = ", pcov2_new[0]**0.5)


#x for m3 for data NEW

x3_new = [ 0,
         *([h1_new-r3_new] * len(m3_h1_new)),
         *([h2_new-r3_new] * len(m3_h2_new)),
         *([h3_new-r3_new] * len(m3_h3_new)),
         *([h4_new-r3_new] * len(m3_h4_new)),
         *([h5_new-r3_new] * len(m3_h5_new)),
         *([h6_new-r3_new] * len(m3_h6_new)),
         *([h7_new-r3_new] * len(m3_h7_new)),
         *([h8_new-r3_new] * len(m3_h8_new)),
         *([h9_new-r3_new] * len(m3_h9_new))
]

error_y3_new=[0.004] * len(x3_new)

#t for m3
time_m3_new = [0, *m3_h1_new, *m3_h2_new, *m3_h3_new, *m3_h4_new, *m3_h5_new, *m3_h6_new, *m3_h7_new, *m3_h8_new, *m3_h9_new]


def get_g3_new(t, g):
    return get_g(m3_new, r3_new, t, g)

popt3_new, pcov3_new = curve_fit(get_g3_new, time_m3_new, x3_new, sigma=error_y3_new)
print("g for m1 for dat new =", popt3_new[0])
print("uncertainty g for m1 for data new = ", pcov3_new[0]**0.5)

print("AVERAGE g:")
print (((popt3_new[0]+popt2_new[0]+popt1_new[0]+popt1_1[0]+popt2_1[0]+popt1_2[0]+popt2_2[0]+popt3_2[0])/8), "+/-", ((pcov1_1[0]**0.5 +pcov2_1[0]**0.5+pcov1_2[0]**0.5 +pcov2_2[0]**0.5+pcov3_2[0]**0.5+pcov1_new[0]**0.5 +pcov2_new[0]**0.5+pcov3_new[0]**0.5)/8))

print("AVERAGE g for data 1:")
print(((popt1_1[0]+popt2_1[0])/2), "+/-", ((pcov1_1[0]**0.5 +pcov2_1[0]**0.5)/2))

print("AVERAGE g for data 2:")
print(((popt1_2[0]+popt2_2[0]+popt3_2[0])/3), "+/-", ((pcov1_2[0]**0.5 +pcov2_2[0]**0.5+pcov3_2[0]**0.5)/3))

print("AVERAGE g for new data:")
print(((popt3_new[0]+popt2_new[0]+popt1_new[0])/3), "+/-", ((pcov1_new[0]**0.5 +pcov2_new[0]**0.5+pcov3_new[0]**0.5)/3))


#PLOT ALL DATA

#plt.plot(time_m1_1, x1_1, 'x')
#plt.plot(time_m2_1, x2_1, 'x')
#plt.plot(time_m1_2, x1_2, 'x')
#plt.plot(time_m2_2, x2_2, 'x')
#plt.plot(time_m3_2, x3_2, 'x')
plt.plot(time_m1_new, x1_new, 'x')
plt.plot(time_m2_new, x2_new, 'x')
plt.plot(time_m3_new, x3_new, 'x')
#plt.plot(x_line, get_g1_1(x_line, *popt1_1), "-k")
#plt.plot(x_line, get_g2_1(x_line, *popt2_1), "-k")
#plt.plot(x_line, get_g1_2(x_line, *popt1_2), "-k")
#plt.plot(x_line, get_g2_2(x_line, *popt2_2), "-k")
#plt.plot(x_line, get_g3_2(x_line, *popt3_2), "-k")
plt.plot(x_line, get_g1_new(x_line, *popt1_new), "-k")
plt.plot(x_line, get_g2_new(x_line, *popt2_new), "-k")
plt.plot(x_line, get_g3_new(x_line, *popt3_new), "-k")
plt.xlabel("Time to fall (seconds)")
plt.ylabel("Height (Meters)")

plt.legend(("Ball 1 from Data","Ball 2 from Data","Ball 3 from Data"))
plt.show()
print(time_m1_new, x1_new,)
print(time_m2_new, x2_new,)
print(time_m3_new, x3_new,)


#---------------------------------------------------------------------------


#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

##KNOWN VALUES IN THE PENDULUM
D = 1.670 #Total length of rod
p = 0.337 #Position of pivot from top
radius = 0.047 #Radius of discs (It's te same for both)
M = 3.67 #Total mass of whole pendulum
m1 = 1 #Mass of top disc
m2 = 1.4 #Mass od bottom disc
mr = M - m1 -m2 #Mass of rod


# Known distances used for finding COM Of pendulum
d1 = 0.107 #position of Mass 1 from the top
dr = 0.8375 #Position of COM of rod from the top (Half the total distance of rod (167.5/2))

#Known distances used for finding moment of inertia
dr0 = 0.311 #Distance between the pivot and COM of rod
d10 = 0.23 #Distance between the pivot and COM of mass1

#Times
times = pd.read_excel('Reversible pendulum all data .xlsx', sheet_name="times")
x1 = times["x1"].dropna(); #x in meters
T_1 = times["t1"].dropna(); #t1 in seconds
T_2 = times["t2"].dropna(); #t2 in seconds

# Create a "figure" and set its aspect ratio. Play with the numbers to make it square, or long or short. 
fig = plt.figure() #figsize=(7,7)
# Here we only want one plot in the figure to a 1 by 1 set of plots and this is number 1. Leave alone for now. 
ax = fig.add_subplot(111)
#ax.set_xlim(0.25, 0.35)
#ax.set_ylim(1.98, 2.02)
# This nest bit does a lot of work and plots a graph with error bars.
ax.errorbar(x1,           # x coordinates
             T_1,              # y coordinates
             yerr = (0.2/50),     # y errors
             xerr = 0.002,
             marker='o',             # marker used is a cicle 'o'. Could be crosses 'x', or squares 's', or 'none'
             markersize = 4,        # marker size
             color='r',          # overall colour I think ( red)
             ecolor='r',         # edge colour for you marker (red)
             markerfacecolor='r', # red
             linestyle='none',       # no line joining markers, could be a line '-', or a dashed line '--'
             capsize=6,              # width of the end bit of the error bars, not too small nor too big please.
             )

ax.errorbar(x1,           # x coordinates
             T_2,              # y coordinates
             yerr = (0.2/50),     # y errors
             xerr = 0.002,
             marker='x',             # marker used is a cicle 'o'. Could be crosses 'x', or squares 's', or 'none'
             markersize = 4,        # marker size
             color='black',          # overall colour I think
             ecolor='black',         # edge colour for you marker
             markerfacecolor='black',
             linestyle='none',       # no line joining markers, could be a line '-', or a dashed line '--'
             capsize=6,              # width of the end bit of the error bars, not too small nor too big please.
             )
plt.xlabel('x1 of movable mass (m) ', fontsize = 15, fontname='Times New Roman' )
plt.ylabel('time (seconds)', fontsize = 15, fontname='Times New Roman')


from scipy import optimize
def func(x,a,b,c,d,e):
    y=(a*x*x)+(b*x)+c+(d*x*x*x)+(e*x*x*x*x)
    return y
parameters, covariance = optimize.curve_fit(func,x1,T_1)
fit_a1= parameters[0]
fit_b1= parameters[1]
fit_c1= parameters[2]
fit_d1= parameters[3]
fit_e1= parameters[4]
print(fit_a1)
print(fit_b1)
print(fit_c1)
print(fit_d1)
print(fit_e1)

fit_y1 = func(x1, fit_a1, fit_b1, fit_c1,fit_d1 , fit_e1)
plt.plot(x1, fit_y1,'-',label = 'Orientation 1', color = 'b', linestyle='-')


from scipy import optimize
def func(x,a,b,c,d,e):
    y=(a*x*x)+(b*x)+c+(d*x*x*x)+(e*x*x*x*x)
    return y
parameters, covariance = optimize.curve_fit(func,x1,T_2)
fit_a1= parameters[0]
fit_b1= parameters[1]
fit_c1= parameters[2]
fit_d1= parameters[3]
fit_e1= parameters[4]
print(fit_a1)
print(fit_b1)
print(fit_c1)
print(fit_d1)
print(fit_e1)

fit_y1 = func(x1, fit_a1, fit_b1, fit_c1,fit_d1 , fit_e1)
plt.plot(x1, fit_y1,'-',label = 'Orientation 1', color = 'b', linestyle='-')




def x1_line(c, g):
    return g*c

results_c=[]

#CHANGE
i = 0
while i < 35:
    x1 = times["x1"].dropna(); #x in meters
    d20 = x1[i] #Distance between the pivot and COM od Mass2
    d2 = 0.337 + d20 #position of mass 2 from top
    
    # Find the Center of Mass of pendulum
    x1 = m1*d1
    x2 = m2*d2
    xr = mr*dr
    X = (x1+x2+xr) / (m1+m2+mr)
    #print ("center of mass of the pendulum", d20, "cm is {}".format(X))
    
    # Find s (distance between pivot and COM of pendulum)
    s =  X - p
    #print ("distance between pivot and COM of pendulum is {}".format(s))
    
    h1 = s #meter
    h2 = 0.995 - s #meter
    
    c = (((T_1[i]**2+T_2[i]**2)/(h1+h2))+((T_1[i]**2-T_2[i]**2)/(h1-h2)))
    
    results_c.append(c)
        
    i += 1

    
popt_g, pcov_g = curve_fit(x1_line, results_c, (8*np.pi**2))
a1 = popt_g[0]
print("g1 =", a1)
print("uncertainty g1 = ", pcov_g[0]**0.5)


def period(l, g):
    2*np.pi*np.sqrt(l/g)


#------------------------------------------------------------------------------

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

D = 1.670 #Total length of rod
p = 0.337 #Position of pivot from top
radius = 0.047 #Radius of discs (It's te same for both)
M = 3.67 #Total mass of whole pendulum
m1 = 1 #Mass of top disc
m2 = 1.4 #Mass od bottom disc
mr = M - m1 -m2 #Mass of rod


# Known distances used for finding COM Of pendulum
d1 = 0.107 #position of Mass 1 from the top
dr = 0.8375 #Position of COM of rod from the top (Half the total distance of rod (167.5/2))

#Known distances used for finding moment of inertia
dr0 = 0.311 #Distance between the pivot and COM of rod
d10 = 0.23 #Distance between the pivot and COM of mass1


#Times
times = pd.read_excel('Reversible pendulum all data .xlsx', sheet_name="large")
x2 = times["x2"].dropna(); #x in meters
T_3 = times["t3"].dropna(); #t1 in seconds
T_4 = times["t4"].dropna(); #t2 in seconds

# Create a "figure" and set its aspect ratio. Play with the numbers to make it square, or long or short. 
fig = plt.figure() #figsize=(7,7)
# Here we only want one plot in the figure to a 1 by 1 set of plots and this is number 1. Leave alone for now. 
ax = fig.add_subplot(111)
#ax.set_xlim(0.25, 0.35)
#ax.set_ylim(1.98, 2.02)
# This nest bit does a lot of work and plots a graph with error bars.
ax.errorbar(x2,           # x coordinates
             T_3,              # y coordinates
             yerr = (0.2/50),     # y errors
             xerr = 0.002,
             marker='o',             # marker used is a cicle 'o'. Could be crosses 'x', or squares 's', or 'none'
             markersize = 4,        # marker size
             color='r',          # overall colour I think ( red)
             ecolor='r',         # edge colour for you marker (red)
             markerfacecolor='r', # red
             linestyle='none',       # no line joining markers, could be a line '-', or a dashed line '--'
             capsize=6,              # width of the end bit of the error bars, not too small nor too big please.
             )

ax.errorbar(x2,           # x coordinates
             T_4,              # y coordinates
             yerr = (0.2/50),     # y errors
             xerr = 0.002,
             marker='x',             # marker used is a cicle 'o'. Could be crosses 'x', or squares 's', or 'none'
             markersize = 4,        # marker size
             color='black',          # overall colour I think
             ecolor='black',         # edge colour for you marker
             markerfacecolor='black',
             linestyle='none',       # no line joining markers, could be a line '-', or a dashed line '--'
             capsize=6,              # width of the end bit of the error bars, not too small nor too big please.
             )
plt.xlabel('x2 of movable mass (m) ', fontsize = 15, fontname='Times New Roman' )
plt.ylabel('time (seconds)', fontsize = 15, fontname='Times New Roman')


from scipy import optimize
def func(x,a,b,c,d,e):
    y=(a*x*x)+(b*x)+c+(d*x*x*x)+(e*x*x*x*x)
    return y
parameters, covariance = optimize.curve_fit(func,x2,T_3)
fit_a1= parameters[0]
fit_b1= parameters[1]
fit_c1= parameters[2]
fit_d1= parameters[3]
fit_e1= parameters[4]
print(fit_a1)
print(fit_b1)
print(fit_c1)
print(fit_d1)
print(fit_e1)

fit_y1 = func(x2, fit_a1, fit_b1, fit_c1,fit_d1 , fit_e1)
plt.plot(x2, fit_y1,'-',label = 'Orientation 1', color = 'b', linestyle='-')


from scipy import optimize
def func(x,a,b,c,d,e):
    y=(a*x*x)+(b*x)+c+(d*x*x*x)+(e*x*x*x*x)
    return y
parameters, covariance = optimize.curve_fit(func,x2,T_4)
fit_a1= parameters[0]
fit_b1= parameters[1]
fit_c1= parameters[2]
fit_d1= parameters[3]
fit_e1= parameters[4]
print(fit_a1)
print(fit_b1)
print(fit_c1)
print(fit_d1)
print(fit_e1)

fit_y1 = func(x2, fit_a1, fit_b1, fit_c1,fit_d1 , fit_e1)
plt.plot(x2, fit_y1,'-',label = 'Orientation 1', color = 'b', linestyle='-')


def x2_line(c, g):
    return g*c

results_c=[]

#CHANGE
i = 0
while i < 10:
    x2 = times["x2"].dropna(); #x in meters
    d20 = x2[i] #Distance between the pivot and COM od Mass2
    d2 = 0.337 + d20 #position of mass 2 from top
    
    # Find the Center of Mass of pendulum
    x2 = m1*d1
    x3 = m2*d2
    xr = mr*dr
    X = (x2+x3+xr) / (m1+m2+mr)
    #print ("center of mass of the pendulum", d20, "cm is {}".format(X))
    
    # Find s (distance between pivot and COM of pendulum)
    s =  X - p
    #print ("distance between pivot and COM of pendulum is {}".format(s))
    
    h1 = s #meter
    h2 = 0.995 - s #meter
    angle = 16*np.pi/180

    c = (((T_3[i]**2+T_4[i]**2)/(h1+h2))+((T_3[i]**2-T_4[i]**2)/(h1-h2)))
    
    results_c.append(c)
        
    i += 1

    
popt_g, pcov_g = curve_fit(x2_line, results_c, (8*np.pi**2))
a1 = popt_g[0]
print("g1 =", a1*(1+((1/16)*angle**2)+(11/3072)*angle**4))
print("uncertainty g1 = ", pcov_g[0]**0.5)


def period(l, g):
    2*np.pi*np.sqrt(l/g)



#T = 2.01
#angle = 16*np.pi/180

#g = 0.9939/((T/(2*np.pi*(1+((1/16)*angle**2)+(11/3072)*angle**4)))**2)
#print(g)


# %%



# %%

# %%

# %%
