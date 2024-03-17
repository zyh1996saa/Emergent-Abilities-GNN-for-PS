import numpy as np


def find_buses_near_selected_bus(selected_bus):
    near_buses = []
    for i in range(case_zj['branch'].shape[0]):
        if int(case_zj['branch'][i,0]) == selected_bus:
            if int(case_zj['branch'][i,1]) not in near_buses:
                near_buses.append(int(case_zj['branch'][i,1]))
        elif int(case_zj['branch'][i,1]) == selected_bus:
            if int(case_zj['branch'][i,0]) not in near_buses:
                near_buses.append(int(case_zj['branch'][i,0]))
    return near_buses
    
if __name__ == "__main__":

    in_bus = [487]
    
    out_bus = [1116,495,490,489,1141,1154]
    
    searched_bus = []
    
    case_zj = {key: np.load('zj2025.npz')[key] for key in np.load('zj2025.npz')}
    
    is_new_bus = True
    
    while is_new_bus:
        is_new_bus = False
        for bus in in_bus:
            near_buses = find_buses_near_selected_bus(bus)
            for near_bus in near_buses:
                if (near_bus not in out_bus) and (near_bus not in in_bus):
                    is_new_bus = True
                    in_bus.append(near_bus)
                    