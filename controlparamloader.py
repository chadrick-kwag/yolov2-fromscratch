import os

class ControlParamLoader:
    configfile='control_params.config'
    weights=None
    lr=None

    def __init__(self, configfile=None):
        if configfile:
            self.configfile = configfile

        if not os.path.exists(self.configfile):
            raise Exception("config file not found")
        
        
    def getconfig(self):

        # check for file existence again

        if not os.path.exists(self.configfile):
            raise Exception("config file not found")
        
        rlines = [ line.rstrip() for line in open(self.configfile, 'r')]

        # the first and second line are important
        firstline = rlines[0]
        secondline = rlines[1]

        # extract weights from firstline
        weights = firstline.rsplit(',')
        weights = list(map(float,weights))
        assert len(weights)==5

        print(weights)

        lr = secondline.rsplit()
        lr = list(map(float,lr))
        assert len(lr)==1
        print(lr)

        return weights, lr


