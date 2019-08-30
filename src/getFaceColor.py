import argparse
import runSteps

#def strToBool(v):
#    if v.lower() in ('yes', 'true', 't', 'y', '1'):
#        return True
#    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
#        return False
#    else:
#        raise argparse.ArgumentError('Boolean value expected. i.e. true, false, yes, no')

ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
#ap.add_argument("-u", "--username", required=True, default="false", help="The Users user name...")
#ap.add_argument("-n", "--name", required=True, help="path and root name of image, i.e. images/doug")
#ap.add_argument("-s", "--step", required=False, default="1", help="step to run processing from")
#ap.add_argument("-p", "--plot", required=False, default="true", help="plot color values at end")
#ap.add_argument("-f", "--fast", required=False, default="false", help="Speed up processing. Does not save intermediate values")
#ap.add_argument("-b", "--benchmark", required=False, default="false", help="Record k means at different steps in the pipeline")
#ap.add_argument("-z", "--baseline", required=False, default="false", help="record baseline values run k means on base image")
#ap.add_argument("-v", "--save", required=False, default="false", help="Save the data to image stats")
ap.add_argument("-u", "--user_id", required=True, default="false", help="The Users user name...")
args = vars(ap.parse_args())

#imageName = args["name"]
#username = args["username"]
#step = int(args["step"])
#shouldPlot = strToBool(args["plot"])
#fast = strToBool(args["fast"])
#benchmark = strToBool(args["benchmark"])
#baseline = strToBool(args["baseline"])
#save = strToBool(args["save"])
failOnError = True
user_id = args["user_id"]

#if not baseline:
#    error = runSteps.run(username, imageName, fast, save, failOnError)

response = runSteps.run2(user_id)
if response is None:
    print('Error....')


