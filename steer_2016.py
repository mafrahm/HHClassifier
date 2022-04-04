import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from IPython.display import FileLink, FileLinks
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical, plot_model
from keras.callbacks import History, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import math
import pickle
from Training import *
from Plotting import *
from GetInputs import *
from RankNetworks import *
from PredictExternal import *
from functions import *

#from TrainModelOnPredictions import *
#from TrainSecondNetwork import *
#from TrainThirdNetwork import *
from ExportModel import *

from clearml import Task


# ignore weird Warnings: "These warnings are visible whenever you import scipy
# (or another package) that was compiled against an older numpy than is installed."
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
# ignore weird Warnings: "These warnings are visible whenever you import scipy
# (or another package) that was compiled against an older numpy than is installed."
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


variations = ['NOMINAL']
merged_str = 'Merged'
input_tag = 'removeQCD_TrainOn30' #_2016
bkgweight = 8
parameters = {
     'layers':[512,512,512],
     'batchsize': 131072, #131072,
     #'classes':{0: ['HH_kl1'], 1: ['TTbar', 'ST'], 2: ['WJets', 'DYJets']},
     'classes':{0: ['HH_kl1'], 1: ['TTbar'], 2: ['ST'], 3: ['WJets', 'DYJets']},
     #'classes':{0: ['HH_kl1'], 1: ['TTbar'], 2: ['ST'], 3: ['WJets', 'DYJets'], 4: ['QCDMu', 'QCDEle']},
     'regmethod': 'dropout',
     #'regmethod': 'L2',
     'regrate':0.50,
     'batchnorm': False,
     'epochs':200, #200, 400
     'learningrate': 0.00050,
     'runonfraction': 1, #0.49, 0.01
     'eqweight': True, #False, TODO differend kinds of reweighting
     'preprocess': 'StandardScaler',
     'sigma': 1.0, #sigma for Gaussian prior (BNN only)
     #'inputdir': '/nfs/dust/cms/user/frahmmat/HHtoWWbbSemiLeptonic/MLtest_numpy',
     'inputdir': '../Input_'+input_tag+'/',             
     #'systvar': variations[ivars],
     'systvar': '',
     'inputsubdir': 'MLInput/', #path to input files: inputdir + systvar + inputsubdir
     'prepreprocess': 'RAW', #for inputs with systematics don't do preprocessing before merging all inputs on one,     #FixME: add prepreprocessing in case one does not need to merge inputs
     'bkgweight': bkgweight,
     'inputtag': 'bkgweightX'+str(bkgweight)+'_'+input_tag
}

tag = dict_to_str(parameters)

classtag = get_classes_tag(parameters)

########## GetInputs ########
for ivars in range(len(variations)):
     merged_str = merged_str+'__'+variations[ivars]
     parameters['systvar'] = variations[ivars]
     # # # # # # Get all the inputs
     # # # # # # # # # ==================
     inputfolder = parameters['inputdir']+parameters['inputsubdir']+parameters['systvar']+'/'+parameters['prepreprocess']+'/'+ classtag
     GetInputs(parameters)
     PlotInputs(parameters, inputfolder=inputfolder, filepostfix='', plotfolder='plots_'+parameters['inputtag']+'/'+parameters['prepreprocess']+'/InputDistributions/'+parameters['systvar']+'/' + classtag)
   
#the inputfolder of the DNN is the outputfolder of the preprocessing     
inputfolder = parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag

MixInputs(parameters, outputfolder=inputfolder, variations=variations, filepostfix='')
SplitInputs(parameters, outputfolder=inputfolder, filepostfix='')
FitPrepocessing(parameters, outputfolder=inputfolder, filepostfix='')
ApplyPrepocessing(parameters, outputfolder=inputfolder, filepostfix='',setid='train')
ApplyPrepocessing(parameters, outputfolder=inputfolder, filepostfix='',setid='test')
ApplyPrepocessing(parameters, outputfolder=inputfolder, filepostfix='',setid='val')
ApplySignalPrepocessing(parameters, outputfolder=inputfolder, filepostfix='')

outputfolder_merged='output_'+parameters['inputtag']+'/'+parameters['preprocess']+'/'+merged_str+'/' + classtag+'/DNN_'+tag
plotfolder = 'plots_'+parameters['inputtag']+'/'+parameters['preprocess']
PlotInputs(parameters, inputfolder=inputfolder, filepostfix='', plotfolder=plotfolder+'/InputDistributions/'+merged_str+'/' + classtag)

#####

task = Task.init(project_name='NN_'+parameters['inputtag'], task_name=dict_to_str(parameters))

# DNN 

outputfolder = 'output_'+parameters['inputtag']+'/'+parameters['preprocess']+'/DNN_'+tag
TrainNetwork(parameters, inputfolder=inputfolder, outputfolder=outputfolder)
PredictExternal(parameters, inputfolder=inputfolder, outputfolder=outputfolder, filepostfix='')
PlotPerformance(parameters, inputfolder=inputfolder, outputfolder=outputfolder, filepostfix='', plotfolder=plotfolder+'/DNN_'+tag, use_best_model=True, usesignals=[0])
#ExportModel(parameters, inputfolder='input/', outputfolder='output/', use_best_model=True)

