import sys
import math

kb = 1.380649E-16 #ergs/kelvin
Ae = 14.137 * 100**2 #cm^2
Ts = 30 #kelvin
bandwidth = 183E3 #Hz

def autocorr_mean ():
    return kb*Ts/Ae * 1E29 #μJy

def autocorr_stdv (int_time):
    return kb*Ts/(Ae*math.sqrt(2*int_time*bandwidth)) * 1E29

def not_autocorr_stdv (int_time):
    return kb*Ts/(2*Ae*math.sqrt(int_time*bandwidth)) * 1E29

if __name__ == "__main__":
    int_time = float(sys.argv[1])
    print("Autocorr distribution: G("+str(autocorr_mean()) +" μJy, " + str(autocorr_stdv (int_time)) +" μJy)")
    print("Not autocorr distribution: G(0 μJy, " + str(not_autocorr_stdv (int_time)) +" μJy)")
