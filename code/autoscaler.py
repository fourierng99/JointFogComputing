from  config import *
class AutoScaler:
    def __init__(self,env):
        self.env = env
        self.max_server = 5

    def calculate_servers(self,request_seq):
        x = max(request_seq)
        volume = x/float(self.max_server)
        vm_seq = []
        for r in request_seq:
            v = float(r)/volume
            if(v> round(v)):
                vm_seq.append(min(round(v+1),self.max_server ))
            else:
                vm_seq.append(min(round(v),self.max_server ))
        
        self.vms = vm_seq
        return vm_seq


asl = AutoScaler("")
asl.calculate_servers([200,312,323,54,23,425,234,756])
print(asl.vms)