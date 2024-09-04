import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

# URI to the Crazyflie to connect to
uri = 'radio://0/80/2M/E7E7E7E7E7'

def print_available_variables(scf):
    toc = scf.cf.log.toc.toc
    for group in toc:
        for variable in toc[group]:
            print(f"{group}.{variable}")

if __name__ == '__main__':
    cflib.crtp.init_drivers()

    # Connect to the Crazyflie
    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        print("Connected to Crazyflie. Available log variables:")
        print_available_variables(scf)
