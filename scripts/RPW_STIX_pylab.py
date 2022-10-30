# -*- coding: utf-8 -*-

# SOLAR ORBITER DATA ANALYSIS LAB
# RPW - STIX========================
# functionalities
# RPW-----
#   * create/plot psd from CDF
#   * frequency drift analysis/ beam veloicty estimation
# STIX -----------
#    * spectrogram creation / bkg removal
#    * spectrogram/lightcurve plotting
#    * combined views of RW/STIX



#package imports
# UTILS
import datetime as dt
from datetime import datetime,time,timedelta
import os
## PLOT
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
## MATH
from scipy.optimize import curve_fit as cfit
import numpy as np


## GUI
from ipywidgets import interact, interactive, widgets, fixed,interact_manual,FloatSlider
try:
    from ipywidgets import Layout
except:
    pass 

## ASTROPY
from astropy.io import fits
from astropy.time.core import Time, TimeDelta
from astropy.table import Table, vstack, hstack
import astropy.units as u

#os.environ["CDF_LIB"] = "~/Documents/cdfpy38/src/lib/"
#os.environ["CDF_LIB"] = "~/Documents/cdf38/src/lib/"
os.environ["CDF_LIB"] = "/home/dpaipa/Documents/cdf38/src/lib/"
from spacepy import pycdf



# CONSTANTS

#date
std_date_fmt = '%d-%b-%Y %H:%M:%S'
simple_date_fmt='%Y-%m-%d'
numeric_date_fmt='%Y-%m-%d %H:%M:%S'
fits_date_fmt="%Y%m%dT%H%M%S"
speed_c_kms = 299792.458
Rs_per_AU = 215.032
km_per_Rs = 695700.
WmHz_per_sfu=1e-22

# rpw indexes
rpw_suggested_freqs_idx =[437,441,442,448,453,458,465,470,477,482,                     
                      493,499,511,519,526,533,
                      538,545,552,559,566,576,588,592,600,612,
                      656,678,696,716,734,741,750,755,505,629,649,
                      673,703,727]
#rpw_suggested_freqs_idx=[ 437,441,442,448,453,458,465,470,
#                         477,482,493,499,511,519,526,533,
#                         538,545,552,559,566,600,
#                         612,678,696,#,576,588,592,649,629,656,673
#                         703,716,727,734,741,750,755]
rpw_idx_hfr=436
rpw_suggested_indexes = np.array(rpw_suggested_freqs_idx)-rpw_idx_hfr

#display_freqs=[0,100,500,1000,2500,5000,10000,15000]
display_freqs=[0,100,500,1000,2000,3500,6000,8000,100000,12000,16000]



## FILE AVAILABILITY  (CHECK)
def stix_get_all_bkg_files(path,verbose=True):

    filelist = []

    for root, dirs, files in os.walk(path):
        if(not"_BKG" in root):
            continue
        for file in files:
            if(not ".fits" in file):
                continue
            #append the file name to the list
            obstime = file.split("_")[3].split("-")[0]
            date = dt.datetime.strptime(obstime,"%Y%m%dT%H%M%S")
            entry= "["+date.strftime("%Y-%m-%d")+"] "+file
            filelist.append(entry)


    #print all the file names
    filelist.sort()
    print(len(filelist),"BKG files found:")
    for name in filelist:
        print(name)

def stix_get_all_aux_files(path,verbose=True):

    filelist = []

    for root, dirs, files in os.walk(path):
        if("_BKG" in root):
            continue
        for file in files:
            if(not ".fits" in file or not "aux" in file):
                continue
            #append the file name to the list
            obstime = file.split("_")[3].split("-")[0]
            date = dt.datetime.strptime(obstime,"%Y%m%d")
            entry= "["+date.strftime("%Y-%m-%d")+"] "+file
            filelist.append(entry)


    #print all the file names
    filelist.sort()
    print(len(filelist),"AUX files found:")
    for name in filelist:
        print(name)

def stix_suggest_bkg_file_for_date(date,rootpath,dt_fmt="%Y-%m-%d %H:%M:%S",suggestions=1):
    date = dt.datetime.strptime(date,dt_fmt)
    filelist = {}
    file_dt_fmt="%Y%m%dT%H%M%S"

    for root, dirs, files in os.walk(rootpath):
        if(not"_BKG" in root):
            continue
        for file in files:
            if(not ".fits" in file):
                continue
            #append the file name to the list
            obstime = file.split("_")[3].split("-")[0]
            dateobs = dt.datetime.strptime(obstime,file_dt_fmt)

            tdelta = (date-dateobs).days
            filelist[os.path.join(root,file)] = np.abs(tdelta)

    sort_files = sorted(filelist.items(),key=lambda x:x[1])
    sort_files = sort_files[:suggestions]
    for i in range(suggestions):
        print("  [",round(sort_files[i][1]),"days] ",(sort_files[i][0]).replace(rootpath,"PATH + "))
    return sort_files

def stix_suggest_bkg_file_for_file(file,rootpath,suggestions=1):
    obstime = file.split("_")[3].split("-")[0]
    fmtd = "%Y%m%dT%H%M%S"
    return stix_suggest_bkg_file(obstime,rootpath,dt_fmt=fmtd,suggestions=suggestions)



def stix_data_in_interval_exists(date_range,rootpath,dt_fmt="%Y-%m-%d %H:%M:%S"):

    date_range = [dt.datetime.strptime(x,dt_fmt) for x in date_range]
    filelist = {
        "totally":[],
        "partially":[]}

    for root, dirs, files in os.walk(rootpath):
        if("_BKG" in root):
            continue
        for file in files:
            if(not ".fits" in file or "aux" in file):
                continue
            #append the file name to the list
            #print(file)
            obstime_1 = file.split("_")[3].split("-")[0]
            obstime_2 = file.split("_")[3].split("-")[1]

            dateinterv = [dt.datetime.strptime(x,"%Y%m%dT%H%M%S") for x in [obstime_1,obstime_2]]

            date1_in_range = date_range[0]>=dateinterv[0] and date_range[0]<=dateinterv[1]
            date2_in_range = date_range[1]>=dateinterv[0] and date_range[1]<=dateinterv[1]

            interv_in_range = date_range[0]<=dateinterv[0] and date_range[1]>=dateinterv[1]

            fileroot=(root+"/"+file).replace(rootpath,"PATH + ")

            if(date1_in_range and date2_in_range):
                filelist["totally"].append(file)
                print(" Totally contained in file:\n  ",
                      fileroot,
                      "\n  File from ",dt.datetime.strftime(dateinterv[0],dt_fmt)," to ",
                      dt.datetime.strftime(dateinterv[1],dt_fmt),"\n")
            elif(date1_in_range or date2_in_range):
                filelist["partially"].append(file)
                print(" Partially contained in file:\n  ",
                      fileroot,
                      "\n  File from ",dt.datetime.strftime(dateinterv[0],dt_fmt)," to ",
                      dt.datetime.strftime(dateinterv[1],dt_fmt),"\n")
            elif(interv_in_range):
                filelist["partially"].append(file)
                print(" Partially contained in file:\n  ",
                      fileroot,
                      "\n  File from ",dt.datetime.strftime(dateinterv[0],dt_fmt)," to ",
                      dt.datetime.strftime(dateinterv[1],dt_fmt),"\n")

    if(len(filelist["totally"])==0 and len(filelist["partially"])==0):
        print("No files containing totally or partially the provided time interval.")

    return filelist


def stix_check_interval_availability(date_range,rootpath,dt_fmt="%Y-%m-%d %H:%M:%S"):
    print("**STIX Science files availability:")
    L1_check = stix_data_in_interval_exists(date_range,rootpath=rootpath,dt_fmt=dt_fmt)
    print("**STIX BKG files availability:")
    BKG_check = stix_suggest_bkg_file_for_date(date_range[0],rootpath=rootpath,dt_fmt=dt_fmt)
    return [L1_check,BKG_check]




def rpw_check_date_availability(date,rootpath,dt_fmt="%Y-%m-%d %H:%M:%S"):
    date = dt.datetime.strptime(date,dt_fmt)
    filelist = []
    simple_dt_fmt="%Y%m%d"

    print("**RPW Science files availability:")
    for root, dirs, files in os.walk(rootpath):
        for file in files:
            if(not ".cdf" in file):
                continue
            #append the file name to the list
            obstime = file.split("_")[3]

            if(date.strftime(simple_dt_fmt)==obstime):

                print("  Contained in file PATH + ",file)
                filelist.append(file)
    return filelist



def check_combined_availability(date_range,rootpath_stix,rootpath_rpw,dt_fmt="%Y-%m-%d %H:%M:%S"):
    stix_info = stix_check_interval_availability(date_range,rootpath_stix,dt_fmt)
    rpw_info=None
    if(date_range[0].split()[0]==date_range[1].split()[0]):
        rpw_info = rpw_check_date_availability(date_range[0],rootpath_rpw,dt_fmt)
    else:
        print("[!] date range includes two different days, two RPW files might be required")
        rpw_info = [rpw_check_date_availability(date_range[0],rootpath_rpw,dt_fmt),
                   rpw_check_date_availability(date_range[1],rootpath_rpw,dt_fmt)]


    return {"STIX":stix_info,"RPW":rpw_info}

#AUX functions


# SOLAR EVENTS CLASS
class solar_event:
    def __init__(self,event_type,times,color=None,linestyle="-",linewidth=2,hl_alpha=0.4,paint_in=None,date_fmt=std_date_fmt):
        self.type = event_type
        #interval,stix_flare,rpw_burst
        try:
            self.start_time = dt.datetime.strptime(times['start'],date_fmt)
        except:
            self.start_time = None
        try:
            self.end_time = dt.datetime.strptime(times['end'],date_fmt)
        except:
            self.end_time = None
        try:
            self.peak_time = dt.datetime.strptime(times['peak'],date_fmt)
        except:
            self.peak_time = None
            
           
        #    self.end_time = times['end'] if  times['end'] else None
        #self.peak_time = times['peak'] if  times['peak'] else None
        self.color = color
        self.linestyle=linestyle
        self.linewidth=linewidth
        self.hl_alpha=hl_alpha
        self.paint_in=paint_in
        if(self.paint_in==None):
            if self.type=="rpw_burst":
                self.paint_in = "rpw"
            elif self.type=="stix_flare":
                self.paint_in = "stix"
            else:
                self.paint_in = "both"
        
        
        
    def paint(self):
        if(self.type=="interval"and self.start_time and self.end_time):
            color = self.color if  self.color else "white"
            plt.axvspan(self.start_time, self.end_time, color=color, alpha=self.hl_alpha)
            plt.axvline(self.start_time,c=color,linestyle=self.linestyle,linewidth=self.linewidth)
            plt.axvline(self.end_time,c=color,linestyle=self.linestyle,linewidth=self.linewidth)
        elif(self.type=="stix_flare"):
            color = self.color if  self.color else "white"
            if(self.start_time):
                plt.axvline(self.start_time,c=color,linestyle="--",linewidth=self.linewidth)
            if(self.peak_time):
                plt.axvline(self.peak_time,c=color,linestyle="-",linewidth=self.linewidth)
        elif(self.type=="rpw_burst" and self.peak_time):
            color = self.color if  self.color else "orange"
            plt.axvline(self.peak_time,c=color,linestyle="-",linewidth=self.linewidth)
        elif(self.type=="marker" and self.peak_time):
            color = self.color if  self.color else "magenta"
            plt.axvline(self.peak_time,c=color,linestyle="-",linewidth=self.linewidth)
    


# RPW data rpw

def rpw_read_hfr_cdf(filepath, sensor=9, start_index=0, end_index=-99):
    
    
    #import datetime

    with pycdf.CDF ( filepath ) as l2_cdf_file:

        frequency = l2_cdf_file[ 'FREQUENCY' ][ : ]  # / 1000.0  # frequency in MHz
        nn = np.size ( l2_cdf_file[ 'Epoch' ][ : ] )
        if end_index == -99:
            end_index = nn
        frequency = frequency[ start_index:end_index ]
        epochdata = l2_cdf_file[ 'Epoch' ][ start_index:end_index ]
        sensor_config = np.transpose (
            l2_cdf_file[ 'SENSOR_CONFIG' ][ start_index:end_index, : ]
        )
        agc1_data = np.transpose ( l2_cdf_file[ 'AGC1' ][ start_index:end_index ] )
        agc2_data = np.transpose ( l2_cdf_file[ 'AGC2' ][ start_index:end_index ] )
        sweep_num = l2_cdf_file[ 'SWEEP_NUM' ][ start_index:end_index ]
        cal_points = (
            l2_cdf_file[ 'FRONT_END' ][ start_index:end_index ] == 1
        ).nonzero ()
    frequency = frequency[ cal_points[ 0 ] ]
    epochdata = epochdata[ cal_points[ 0 ] ]
    sensor_config = sensor_config[ :, cal_points[ 0 ] ]
    agc1_data = agc1_data[ cal_points[ 0 ] ]
    agc2_data = agc2_data[ cal_points[ 0 ] ]
    sweep_numo = sweep_num[ cal_points[ 0 ] ]
    ssweep_num = sweep_numo
    timet = epochdata

    # deltasw = sweep_numo[ 1:: ] - sweep_numo[ 0:np.size ( sweep_numo ) - 1 ]
    deltasw = abs ( np.double ( sweep_numo[ 1:: ] ) - np.double ( sweep_numo[ 0:np.size ( sweep_numo ) - 1 ] ) )
    xdeltasw = np.where ( deltasw > 100 )
    xdsw = np.size ( xdeltasw )
    if xdsw > 0:
        xdeltasw = np.append ( xdeltasw, np.size ( sweep_numo ) - 1 )
        nxdeltasw = np.size ( xdeltasw )
        for inswn in range ( 0, nxdeltasw - 1 ):
            # sweep_num[ xdeltasw[ inswn ] + 1:xdeltasw[ inswn + 1 ] ] = sweep_num[
            # xdeltasw[ inswn ] + 1:xdeltasw[ inswn + 1 ] ] + \
            # sweep_numo[ xdeltasw[ inswn ] ]
            sweep_num[ xdeltasw[ inswn ] + 1:xdeltasw[ inswn + 1 ] + 1 ] = sweep_num[
                                                                           xdeltasw[ inswn ] + 1:xdeltasw[
                                                                                                     inswn + 1 ] + 1 ] + \
                                                                           sweep_numo[ xdeltasw[ inswn ] ]
    sens0 = (sensor_config[ 0, : ] == sensor).nonzero ()[ 0 ]
    sens1 = (sensor_config[ 1, : ] == sensor).nonzero ()[ 0 ]
    print("  sensors: ",np.shape(sens0),np.shape(sens1))
    psens0 = np.size ( sens0 )
    psens1 = np.size ( sens1 )
    timet_ici=[]

    if (np.size ( sens0 ) > 0 and np.size ( sens1 ) > 0):
        agc = np.append ( np.squeeze ( agc1_data[ sens0 ] ), np.squeeze ( agc2_data[ sens1 ] ) )
        frequency = np.append ( np.squeeze ( frequency[ sens0 ] ), np.squeeze ( frequency[ sens1 ] ) )
        sens = np.append ( sens0, sens1 )
        timet_ici = np.append ( timet[ sens0 ], timet[ sens1 ] )
    else:
        if (np.size ( sens0 ) > 0):
            agc = np.squeeze ( agc1_data[ sens0 ] )
            frequency = frequency[ sens0 ]
            sens = sens0
            timet_ici = timet[ sens0 ]
        if (np.size ( sens1 ) > 0):
            agc = np.squeeze ( agc2_data[ sens1 ] )
            frequency = frequency[ sens1 ]
            sens = sens1
            timet_ici = timet[ sens1 ]
        if (np.size ( sens0 ) == 0 and np.size ( sens1 ) == 0):
            print('  no data at all ?!?')
            V = (321)
            V = np.zeros ( V ) + 1.0
            time = np.zeros ( 128 )
            sweepn_HFR = 0.0
    #           return {
    #               'voltage': V,
    #               'time': time,
    #               'frequency': frequency,
    #               'sweep': sweepn_HFR,
    #               'sensor': sensor,
    #           }
    ord_time = np.argsort ( timet_ici )
    timerr = timet_ici[ ord_time ]
    sens = sens[ ord_time ]
    agc = agc[ ord_time ]
    frequency = frequency[ ord_time ]
    maxsweep = max ( sweep_num[ sens ] )
    minsweep = min ( sweep_num[ sens ] )
    sweep_num = sweep_num[ sens ]

    V1 = np.zeros ( 321 ) - 99.
    V = np.zeros ( 321 )
    freq_hfr1 = np.zeros ( 321 ) - 99.
    freq_hfr = np.zeros ( 321 )
    time = 0.0
    sweepn_HFR = 0.0
    # ind_freq = [(frequency - 0.375) / 0.05]
    ind_freq = [ (frequency - 375.) / 50. ]
    ind_freq = np.squeeze ( ind_freq )
    ind_freq = ind_freq.astype ( int )
    for ind_sweep in range ( minsweep, maxsweep + 1 ):
        ppunt = (sweep_num == ind_sweep).nonzero ()[ 0 ]
        xm = np.size ( ppunt )
        if xm > 0:
            V1[ ind_freq[ ppunt ] ] = agc[ ppunt ]
            freq_hfr1[ ind_freq[ ppunt ] ] = frequency[ ppunt ]
            # print(frequency[ppunt])
        if np.max ( V1 ) > 0.0:
            V = np.vstack ( (V, V1) )
            freq_hfr = np.vstack ( (freq_hfr, freq_hfr1) )
            sweepn_HFR = np.append ( sweepn_HFR, sweep_num[ ppunt[ 0 ] ] )
        V1 = np.zeros ( 321 ) - 99
        freq_hfr1 = np.zeros ( 321 )  # - 99
        if xm > 0:
            time = np.append ( time, timerr[ min ( ppunt ) ] )
    # sys.exit ( "sono qui" )
    V = np.transpose ( V[ 1::, : ] )
    time = time[ 1:: ]
    sweepn_HFR = sweepn_HFR[ 1:: ]
    freq_hfr = np.transpose ( freq_hfr[ 1::, : ] )
    return {
        'voltage': V,
        'time': time,
        'frequency': freq_hfr,
        'sweep': sweepn_HFR,
        'sensor': sensor,
    }


# RPW get data object
def rpw_get_data(file,sensor=9):
    # data read
    print("Extracting info:")
    infos = os.path.basename(file).split("-")[0].split("_")
    #dts = [dt.datetime.strptime(x,"%Y%m%dT%H%M%S") for x in os.path.basename(pathfile).split("_")[3].split("-")]
    #dts = [dt.datetime.strftime(x,dt_fmt)for x in dts]
    txt_type = "{} {}".format(infos[2] ,infos[1]).upper() 
    
    #output  dict
    print("  File: ",os.path.basename(file))
    print("  Type: ",txt_type)
    return rpw_read_hfr_cdf(file,sensor=sensor)


    

# RPW filter frequencies
def rpw_select_freq_indexes(frequency,**kwargs):#,freq_col=0,proposed_indexes=None):
    #indexes of frequencies different from 0 or -99 (column 0 in frequency matrix)
    
    fcol = kwargs["freq_col"]
    freq_nozero = np.where(frequency.T[fcol]>0)[0]
    
    selected_freqs = freq_nozero
    
    dfreq = np.array(frequency[selected_freqs,fcol])
    dfreq = dfreq[1:]-dfreq[:-1]
    
    #print("nz",dfreq)
    if kwargs["which_freqs"]=="both":
        selected_freqs = [ j  for j in kwargs["proposed_indexes"] if j in freq_nozero]
        
    if(not kwargs["freq_range"]==None):
        #print(frequency[selected_freqs,fcol])
        selected_freqs = [selected_freqs[j] for j in range(len(selected_freqs)) if np.logical_and(frequency[selected_freqs[j],fcol] <= kwargs["freq_range"][1], frequency[selected_freqs[j], fcol]>=kwargs["freq_range"][0])]
    
    return selected_freqs,frequency[selected_freqs,fcol],dfreq

# create PSD from rpw data object
def rpw_create_PSD(data,freq_range=None,date_range=None,freq_col=0,proposed_indexes=rpw_suggested_indexes,which_freqs="both",rpw_bkg_interval=None):
    # return,x,y,z
    
    time_data = data["time"]
    date_idx = np.arange(len(time_data))
    start_date,end_date = time_data[0],time_data[-1]
    # when date range provided,select time indexes
    if(date_range):
        
        start_date = dt.datetime.strptime(date_range[0], std_date_fmt)
        end_date = dt.datetime.strptime(date_range[1], std_date_fmt)

        date_idx = np.array( np.where( np.logical_and(time_data<=end_date,time_data>=start_date))[0] ,dtype=int)

        if(len(date_idx)==0):
            print("  RPW Error! no data in between provided date range")
            return
    print("  data cropped from ",start_date," to ",end_date)
    # define time axis
    date_idx = np.array(date_idx)
    t_axis = time_data[date_idx]
    
    #define energy axis
    freq_ = data['frequency']
  
    freq_idx,freq_axis,dfreq = rpw_select_freq_indexes(freq_,freq_col=freq_col,freq_range=freq_range,
                                         proposed_indexes=proposed_indexes,which_freqs=which_freqs)
    freq_idx = np.array(freq_idx)
    print("  Selected frequencies [kHz]: ",*freq_axis.astype(int))
    
    # selecting Z axis (cropping)
    z_axis= data["voltage"][:,date_idx]
    z_axis = z_axis[freq_idx,:]
    
    mn_bkg=None
    
# BKG subtraction (approx) if needed
    if rpw_bkg_interval :
        print("  Creating mean bkg from ",rpw_bkg_interval[0]," to ",rpw_bkg_interval[1],"...")
        start_bkg = dt.datetime.strptime(rpw_bkg_interval[0],std_date_fmt)
        end_bkg = dt.datetime.strptime(rpw_bkg_interval[1],std_date_fmt)
        idx_in = [j for j in range(len(t_axis)) if np.logical_and(t_axis[j]>=start_bkg,t_axis[j]<=end_bkg)]

        mn_bkg = np.mean(z_axis[:,idx_in],axis=1)
        mn_bkg = np.array([mn_bkg for i in range(np.shape(z_axis)[1])]).T
        mn_bkg = mn_bkg.clip(0,np.inf)
        
        z_axis=np.clip(z_axis-mn_bkg,1e-16,np.inf)
        
        print("  bkg done.")

    
    
    
    return_dict = {
        "t_idx":date_idx,
        "freq_idx":freq_idx,
        "time":t_axis,
        "frequency":freq_axis,
        "v":z_axis,
        "df":dfreq,
        "bkg":mn_bkg
    }
    
    return return_dict
# plot PRW psd object
#def rpw_plot_psd(psd,logscale=True,colorbar=True,cmap="jet",t_format="%H:%M:%S",ax=None,
 #           axis_fontsize=13,xlabel=True,frequency_range=None):

  #  t,f,z=psd["time"],psd["frequency"],psd["v"]
 #   if(logscale):
  #      z = np.log10(z)


#    ax = ax if ax else plt.gca()
#    ax.xaxis.set_major_formatter(mdates.DateFormatter(t_format))

    #mn = np.mean(z[:,0:1500],axis=1)
    #bckg_ = np.array([mn for i in range(np.shape(z)[1])]).T
    #z_=np.clip(z-bckg_,1e-16,np.inf)

#    cm= ax.pcolormesh(t,f,z,shading="auto",cmap=cmap)
#    if(colorbar):
#        plt.colorbar(cm,label="$Log_{10}$ PSD (V)")
#    ax.set_yscale('log')
#    ax.set_yticks([], minor=True)
#    ax.set_yticks([x  for x in display_freqs if np.logical_and(x<=f[-1],x>=f[0])])
    #ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    #plt.ticklabel_format(axis='y', style='sci',scilimits=(0,0))
#    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y/1000.),1)))).format(y/1000.)))
#    
#    if(frequency_range):
#        plt.ylim(max(frequency_range[0],np.min(f)),min(frequency_range[1],np.max(f)))     
#    if(xlabel):
#        plt.xlabel("start time: "+t[0].strftime(std_date_fmt),fontsize=axis_fontsize)
#    plt.ylabel("Frequency [MHz]",fontsize=axis_fontsize)
#    
#    return ax

def rpw_plot_psd(psd,logscale=True,colorbar=True,cmap="jet",t_format="%H:%M",ax=None,date_range=None,
            axis_fontsize=13,xlabel=True,frequency_range=None):

    t,f,z=psd["time"],psd["frequency"],psd["v"]
    if(logscale):
        z = np.log10(z)


    ax = ax if ax else plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter(t_format))
    
    if(date_range):
        dt_fmt_ = "%d-%b-%Y %H:%M:%S"
        date_range=[datetime.strptime(x,dt_fmt_) for x in date_range]
        plt.xlim(*date_range)

    cm= ax.pcolormesh(t,f,z,shading="auto",cmap=cmap)
    if(colorbar):
        plt.colorbar(cm,label="$Log_{10}$ PSD (V)")

    
    
    ax.set_yscale('log')
    ax.set_yticks([], minor=True)
    ax.set_yticks([x  for x in display_freqs if np.logical_and(x<=f[-1],x>=f[0])])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y/1000.),1)))).format(y/1000.)))
    
    if(frequency_range):
        plt.ylim(max(frequency_range[0],np.min(f)),min(frequency_range[1],np.max(f)))
    
    
    if(xlabel):
        plt.xlabel("start time: "+t[0].strftime(std_date_fmt),fontsize=axis_fontsize)
    plt.ylabel("Frequency [MHz]",fontsize=axis_fontsize)
    
    return ax
# plot rpw curves
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def rpw_plot_curves(rpw_psd,savename=None,
                      dt_fmt=std_date_fmt,title=None,ax=None,
                      date_range=None,legend=True,fill_nan=True,lcolor=None,lw=1,ls="-",
                      freqs=None,ylogscale=True,smoothing_pts=None,bias_multiplier=1):
    
    color_list = ["red","dodgerblue","limegreen","orange","cyan","magenta"]

    t,f,z=rpw_psd["time"],rpw_psd["frequency"],rpw_psd["v"]
    #if(logscale):
    #    z = np.log10(z)


    ax = ax if ax else plt.gca()
    
    
 
    myFmt = mdates.DateFormatter(dt_fmt)
    
    #if(fill_nan):
    #    cts_data=np.nan_to_num(cts_data,nan=0)
    
    plot_groups = []
    
    for sel_freq in freqs:
        idx_close = np.argmin(np.abs(rpw_psd["frequency"]-sel_freq))
        close_freq = rpw_psd["frequency"][idx_close]
        close_intensity = z[idx_close,:]
        if(smoothing_pts):
            #moving average smoothing
            close_intensity = smooth(close_intensity,smoothing_pts)
        
        
        plot_groups.append([close_freq,close_intensity])
        
        
    if not lcolor:
        lcolor=color_list[:len(plot_groups)]
    elif len(lcolor)<len(plot_groups):
        print("[!] color list length do not match the number of energy bins plotted. Using default instead")
        lcolor=color_list[:len(plot_groups)]

    lims = [0,0]
    for g in range(len(plot_groups)):
        pg = plot_groups[g]
        
        if(bias_multiplier):
            plot_y = pg[1]*bias_multiplier**(len(plot_groups)-g)
            ax.plot(t,plot_y,label="{} kHz".format(int(pg[0])),c=lcolor[g],lw=lw,ls=ls)
            ax.plot([t[0],t[-1]],[np.min(plot_y),np.min(plot_y)],c=lcolor[g],lw=0.5,ls=ls)
            if(np.max(plot_y)>lims[1] or lims[1]==0):
                lims[1] = 2*np.max(plot_y)
            if(np.min(plot_y)<lims[0]or lims[0]==0):
                lims[0] = 5*np.min(plot_y)
        else:
            
            ax.plot(t,pg[1],label="{} kHz".format(int(pg[0])),c=lcolor[g],lw=lw,ls=ls)
            if(np.max(pg[1])>lims[1]or lims[1]==0):
                lims[1] = 2*np.max(pg[1])
            if(np.min(pg[1])<lims[0]or lims[0]==0):
                lims[0] = 5*np.min(pg[1])
    if(legend):
        plt.legend(loc=2)
    if(ylogscale):
        plt.yscale("log")
    plt.ylim(lims[0],lims[1])
    return ax
     
 # STIX data read

def stix_create_counts(pathfile, is_bkg=False,time_arr=None,correct_flight_time=False):
    
    print("Extracting info: ")
    
    infos = os.path.basename(pathfile).split("-")[0].split("_")
    #dts = [dt.datetime.strptime(x,"%Y%m%dT%H%M%S") for x in os.path.basename(pathfile).split("_")[3].split("-")]
    #dts = [dt.datetime.strftime(x,dt_fmt)for x in dts]
    txt_type = "{} {}".format(infos[2] ,infos[1]).upper() 
    
    print("  File: ",os.path.basename(pathfile))
    print("  Type: ",txt_type,"BKG" if is_bkg else "")
    #output  dict
    return_dict = {}
    #fits info
    hdulist = fits.open(pathfile)
    header = hdulist[0].header
    earth_sc_delay = header["EAR_TDEL"] if correct_flight_time else 0
    
    print("  Obs. elapsed time: ",round((Time(header["DATE_END"])-Time(header["DATE_BEG"])).to(u.s).value/60,2),"minutes")

    
    data = Table(hdulist[2].data)
    
    #sum over all detectors and pixels (optional)
    data_counts = np.sum(data['counts'],axis=(1,2))
    # get cts_per_sec
    n_energies = len(hdulist[3].data["channel"])
    print ("  Energy channels extracted: ",n_energies)
    #normalise by time_bin duration ("timedel" keyword)
    if is_bkg and np.shape(data['timedel'])[0]>1:
        data_counts = np.mean(data_counts,axis=0)
        timedel = np.mean(data['timedel'])
        data_counts_per_sec = np.reshape(data_counts/timedel,(n_energies))
    else:
        data_counts_per_sec = np.reshape(data_counts/data['timedel'],(n_energies)) if is_bkg else data_counts/data['timedel'].reshape(-1,1)
    
    # for bakground create array of constant bkg cts/sec value per energy bin
    if is_bkg:
        bkg_arr = []
        for i in range(len(time_arr)):
            bkg_arr.append(data_counts_per_sec)
            
            
            
        return_dict = {"time":time_arr,
                       "counts_per_sec":bkg_arr
        }
    # for L1 images, return energy info , cts/sec/bin, time array
    else:
        energies = Table(hdulist[3].data)
        max_e = np.max(energies["e_low"])
        mean_energy = [(min(max_e+1,e_high)+e_low)/2 for chn,e_low,e_high in hdulist[3].data]
        
        data_time = Time(header['date_obs']) + TimeDelta(data['time'] * u.s)+TimeDelta(earth_sc_delay * u.s)
        data_time = [t.datetime for t in data_time]
    # counts object, input  for plotting and spectral analysis routines
        return_dict = {"time":data_time,
                   "counts_per_sec":data_counts_per_sec,
                   "energy_bins":energies,
                   "mean_energy":mean_energy}
    return return_dict






def stix_remove_bkg_counts(pathfile,pathbkg,correct_flight_time = False):
    
    #import L1 data
    data_L1 = stix_create_counts(pathfile)
    #import BKG data
    data_BKG = stix_create_counts(pathbkg,is_bkg=True, time_arr=data_L1["time"],correct_flight_time =correct_flight_time )
    
    #subtract background 
    data_counts_per_sec_nobkg = data_L1["counts_per_sec"]-data_BKG["counts_per_sec"]
    
    #create bkg spectrum
    bkg_count_spec=data_BKG["counts_per_sec"][0]
    
    
    # replace ctc/secinfo with corrected info
    return_dict = data_L1.copy()
    return_dict["counts_per_sec"] = data_counts_per_sec_nobkg
    return_dict["background"]=bkg_count_spec
    
    return return_dict

def stix_combine_files(filenames,bkgfile=None,correct_flight_time=False):
    if bkgfile:
        return stix_combine_counts([stix_remove_bkg_counts(x,bkgfile) for x in filenames])
    else:    
        return stix_combine_counts([sololab.stix_create_counts(x,correct_flight_time=correct_flight_time) for x in filenames])

def stix_combine_counts(allcounts):

    allcounts.sort(key=lambda x: x["time"][0], reverse=False)
    
    eranges = [x["energy_bins"] for x in allcounts]
    same_eranges = all(len(x)==len(eranges[0]) for x in eranges)
    
    
    # merging conditions
    same_energy_bins = same_eranges
    
    
    if(not same_eranges):
        print("Warning! count objects to merge do not have the same energy bins")
        print("   number of bins:")
        print("   ",[len(x) for x in eranges])
    else:
        for i in allcounts[0]["energy_bins"]["channel"]:
            all_chan = [(allcounts[j]["energy_bins"]["e_low"][i],allcounts[j]["energy_bins"]["e_high"][i]) for j in range(len(allcounts))]
            same_echan = all(x==all_chan[0] for x in all_chan)
            if(not same_echan):
                print("Warning! count objects to merge do not have the same energy bins")
                print("   inconsistent channel energies: channel ",i)
                print("   energy limits:",all_chan)
                same_energy_bins=False
    timedelts = [(allcounts[a]["time"][0]-allcounts[a-1]["time"][-1]).seconds for a in range(1,len(allcounts))]
    print("Time gaps between files in seconds (negatives mean overlapping):",timedelts)
    
    if(same_energy_bins):
        new_time = []
        new_cts_per_sec = []
        energy_bins = allcounts[0]["energy_bins"]
        mean_e = allcounts[0]["mean_energy"]
        for i in range(len(allcounts)):
            cts_ = allcounts[i]
            time_=cts_["time"]
            for t in range(len(time_)):
                if(len(new_time)==0 or (time_[t]>new_time[-1])):
                    new_time.append(time_[t])
                    new_cts_per_sec.append(cts_["counts_per_sec"][t,:])
        
    
    
        new_cts_obj={
        "time":new_time,
        "energy_bins":energy_bins,
        "mean_energy":mean_e,
        "counts_per_sec":np.array(new_cts_per_sec),
        }
        return new_cts_obj
    else: 
        print("No return: exception cases for energy bins unconsistency are still under development")




def stix_plot_spectrogram(counts_dict,savename=None,colorbar=True,
                      xfmt=" %H:%M",title=None,cmap="jet",fill_nan=True,
                      date_range=None,energy_range=None,x_axis=False,ax=None,
                      logscale=True,ylogscale=False,**kwargs):
    # date_ranges param is used for visualizing delimiters for date range selection of the 
    # background and sample pieces (interactive plotting)
    # date_ranges = [[bkg_initial, bkg_final],[smpl_initial, smpl_final]]
    
    plot_time = counts_dict["time"]
    cts_per_sec = counts_dict["counts_per_sec"]
    energies = counts_dict ["energy_bins"]
    mean_e = counts_dict["mean_energy"]
    ax = ax if ax else plt.gca()
    
    
    myFmt = mdates.DateFormatter(xfmt)
    ax.xaxis.set_major_formatter(myFmt)

    cts_data = np.log10(cts_per_sec, out=np.zeros_like(cts_per_sec), where=(cts_per_sec>0)).T if logscale else cts_per_sec.T
    if(fill_nan):
        cts_data=np.nan_to_num(cts_data,nan=0)
    cm= plt.pcolormesh(plot_time,mean_e,cts_data,shading="auto",cmap=cmap,vmin=0)
    if(colorbar):
        cblabel = "$Log_{10}$ Counts $s^{-1}$" if logscale else "Counts $s^{-1}$"
        plt.colorbar(cm,label=cblabel)
    if(x_axis):
        plt.xlabel("start time "+date_ranges[0],fontsize=14)
    plt.ylabel('Energy bins [KeV]',fontsize=14)
    if(energy_range):
        plt.ylim(*energy_range)
    if(title):
        plt.title(title)
    if(date_range):
        dt_fmt_ = "%d-%b-%Y %H:%M:%S"
        date_range=[datetime.strptime(x,dt_fmt_) for x in date_range]
        plt.xlim(*date_range)
    if(ylogscale):
        plt.yscale('log')
    #return fig, axes
    if(savename):
        plt.savefig(savename,bbox_inch="tight")
    
    
    return ax


def stix_plot_bkg(l1_counts,ax=None):
    
    ax= ax if ax else plt.gca()
    if not"background" in l1_counts.keys():
        print("This L1 counts object has no removed background.")
        return ax
    
    energies = l1_counts["mean_energy"]
    bkg_counts = l1_counts["background"]
    ax.plot(energies,bkg_counts)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Energy[kev]")
    ax.set_ylabel("Counts per second")
    ax.axvline(31,c="r",ls="--")
    ax.axvline(81,c="r",ls="--",label="Callibration lines \n  31 and 81 kev")
    ax.grid()
    ax.legend(fontsize=12)

    
def stix_plot_counts(counts_dict,savename=None,
                      dt_fmt=std_date_fmt,title=None,e_range=None,ax=None,
                      date_range=None,legend=True,fill_nan=True,lcolor=None,lw=1,ls="-",
                      integrate_bins=None,zlogscale=True,ylogscale=True):
    
    color_list = ["red","blue","green","orange","cyan","magenta"]

        
    
    
    #get data   
    plot_time = counts_dict["time"]
    energies = counts_dict ["energy_bins"]
    mean_e = np.array(counts_dict["mean_energy"])
    cts_per_sec = counts_dict["counts_per_sec"]
    cts_data = cts_per_sec #np.log10(cts_per_sec, out=np.zeros_like(cts_per_sec), where=(cts_per_sec>0)) if zlogscale else cts_per_sec
    
    myFmt = mdates.DateFormatter(dt_fmt)
    
    if(fill_nan):
        cts_data=np.nan_to_num(cts_data,nan=0)
    
    
    # select data
    if(e_range!=None):
        e_idx = np.logical_and(mean_e>=e_range[0],mean_e<=e_range[1])
        #print(e_idx)
 
        cts_data = cts_data[:,e_idx]
        energies = energies[e_idx]
        
        mean_e = mean_e[e_idx]
    if(date_range!=None):
        date_range=[datetime.strptime(x,dt_fmt) for x in date_range]
        d_idx = np.array([True if np.logical_and(x>=date_range[0],x<=date_range[1]) else False for x in plot_time])
        #print(d_idx)
        cts_data = cts_data[d_idx,:]
    
        #plot_time = plot_time[d_idx]
        plot_time = [i for (i, v) in zip(plot_time, d_idx) if v]
     
    
    
    
    plot_groups = []
    
    if(integrate_bins!=None):
        for e_bin in integrate_bins:

            e_idx = np.logical_and(mean_e>=e_bin[0],mean_e<=e_bin[1])
            
            cts_per_sec_g = cts_data[:,e_idx]
            energies_g = energies[e_idx]
            mean_e_g= mean_e[e_idx]
            
            energy_g = [energies_g[0]["e_low"],energies_g[-1]["e_high"]]
            m_energy_g = np.mean([energies_g[0]["e_low"],energies_g[0]["e_high"]])
            cts_sec_g = np.sum(cts_per_sec_g,axis=1)
            
            
            plot_groups.append([cts_sec_g,energy_g,m_energy_g])
    else:
        for e in range(len(energies)):
            plot_groups.append([cts_per_sec[:,e],[energies[e]["e_low"],energies[e]["e_high"]],mean_e[e]])
        
    if not lcolor:
        lcolor=color_list[:len(plot_groups)]
    elif len(lcolor)<len(plot_groups):
        print("[!] color list length do not match the number of energy bins plotted. Using default instead")
        lcolor=color_list[:len(plot_groups)]
    
    ax = ax if ax else plt.gca()
    #ax.xaxis.set_major_formatter(myFmt)
    
    lims = [10,100]
    for g in range(len(plot_groups)):
        pg = plot_groups[g]
        
        ax.plot(plot_time,pg[0]+1,label="{} - {} keV".format(int(pg[1][0]),int(pg[1][1])),c=lcolor[g],lw=lw,ls=ls)
        if(np.max(pg[0]+1)>lims[1]):
            lims[1] = 2*np.max(pg[0]+1)
    if(legend):
        plt.legend()
    if(ylogscale):
        #plt.ylim(0.5,np.max(cts_per_sec))
        plt.yscale("log")
    plt.ylim(lims[0],lims[1])
    return ax
     

    
# Combined views
def stix_rpw_combinedQuickLook(l1_cts,rpw_psd,energy_range=[4,28],frequency_range=[400,17000],energy_bins=[[4,12],[16,28]],
                               frequencies=[500,3500,13000],stix_cmap="bone",rpw_cmap="bone",cmap=None,date_fmt="%H:%M",
                               date_range=None,stix_ylogscale=False,smoothing_points=5,stix_lcolor=None,rpw_lcolor=None,rpw_units="watts",
                               figsize=(15,7),mode="overlay",curve_overlay="both",rpw_plot_bias=False,curve_lw=2,fontsize=13,savename=None):
    #font = { #'family' : 'normal',
        #'weight' : 'normal',
    #    'size'   : 15}
    rpw_psd_cp=rpw_psd.copy()
    rpw_ylabel = "I "
    if rpw_units=="watts":
        rpw_ylabel+="[W m$^{-2}$ Hz$^{-1}$]"
    elif rpw_units=="sfu":
        rpw_psd_cp["v"]= rpw_psd_cp["v"]/1e-22
        rpw_ylabel += " [SFU]"
        
        
    
    plt.rcParams.update({'font.size': fontsize})

    # palette for spectrograms
    stix_cmap = cmap if cmap else stix_cmap
    rpw_cmap = cmap if cmap else rpw_cmap
    
    
     #set date range(str): if not provided, then, use max possible
    drange=None
    
    if date_range:
        drange = date_range
    else:
        drange = [np.max([l1_cts["time"][0],rpw_psd_cp["time"][0]]),
              np.min([l1_cts["time"][-1],rpw_psd_cp["time"][-1]])]
        drange = [datetime.strftime(x,std_date_fmt) for x in drange]
    
    xlims =[datetime.strptime(x,std_date_fmt) for x in drange]
    
    myFmt = mdates.DateFormatter(date_fmt)
    fig=plt.figure(figsize=figsize,dpi=250)
    
    
    if(mode in ["overlay","spectrograms"]):
        #CHANGe
        ax1 = fig.add_subplot(211)
        rpw_plot_psd(rpw_psd_cp,xlabel=False,frequency_range=frequency_range,date_range=drange,
                             cmap=rpw_cmap,t_format="%H:%M",ax=ax1)
        if(curve_overlay in ["both","rpw"] and mode=="overlay"):
            
            ax1b = ax1.twinx()
            
            #CHANGE
            multip=round(5*10**np.interp(len(frequencies),[2,7],[3.5,2]),-2) if rpw_plot_bias else None
            rpw_plot_curves(rpw_psd_cp,freqs=frequencies,ax=ax1b,lw=curve_lw,smoothing_pts=smoothing_points,
                            bias_multiplier=multip)
            ax1b.get_yaxis().set_ticks([])
            ax1b.set_yticklabels([])
        ax1.invert_yaxis()
        ax2 = fig.add_subplot(212)
        stix_plot_spectrogram(l1_cts,ax=ax2,cmap=stix_cmap,energy_range=energy_range,date_range=drange)
        if(stix_ylogscale):
            plt.yscale('log')
        if(curve_overlay in ["both","stix"] and mode=="overlay"):
            ax3=ax2.twinx()
            stix_plot_counts(l1_cts,integrate_bins=energy_bins,
                                 lcolor=stix_lcolor,ax=ax3,lw=curve_lw)
    elif(mode=="curves"):
        stix_rows=max(1,int(len(frequencies)/2)-1)
        rows = stix_rows+len(frequencies)
        ax0 = plt.subplot2grid((rows, 1), (rows-stix_rows, 0), rowspan=stix_rows)
        stix_plot_counts(l1_cts,integrate_bins=energy_bins,
                                 lcolor=stix_lcolor,ax=ax0,lw=curve_lw,date_range=drange)
        ax0.xaxis.set_major_formatter(myFmt)
        ax0.grid()
        ax0.set_xlim(xlims)
        
        freq_axs=[]
        for i in range(len(frequencies)):
            freq_axs.append(plt.subplot2grid((rows, 1), (i, 0)))
            rpw_plot_curves(rpw_psd_cp,freqs=[frequencies[i]],ax=freq_axs[-1],lw=curve_lw,smoothing_pts=smoothing_points,lcolor=["k"],
                            bias_multiplier=None)
            if(i==0):
                freq_axs[-1].xaxis.tick_top()
                
                freq_axs[-1].xaxis.set_major_formatter(myFmt)
                freq_axs[-1].set_ylabel(rpw_ylabel)
                
            else:
                freq_axs[-1].set_xticklabels([])
                
            freq_axs[-1].set_xlim(xlims)
            freq_axs[-1].grid()
            
            

            #plt.tight_layout()
            #plt.savefig('grid_figure.pdf')
    #plt.xlim()
    return fig

#["red","dodgerblue","limegreen","magenta"]


def rpw_stix_combined_view(stx_cts,rpw_psd,date_range=None,dt_fmt=std_date_fmt,figsize=[15,9],
                          rpw_freq_range=None,stix_energy_range=None,invert_rpw_axis=True,markers={},markerwidth=1.5,
                           cbars=True,stix_logscale=True,
                          common_interval=True,stix_cmap="jet",rpw_cmap="jet",events=[]):

    
    # select time range datetime
    time_interval = [dt.datetime.strptime(x,dt_fmt) for x in date_range]
    
    common_interval = [max(np.min(stx_cts["time"]),np.min(rpw_psd["time"])),
                      min(np.max(stx_cts["time"]),np.max(rpw_psd["time"]))]
    if(common_interval):
        time_interval = [max(common_interval[0],time_interval[0]),
                        min(common_interval[1],time_interval[1])]
        print("Time axis constrained to common time interval...")
    
    new_date_range = [dt.datetime.strftime(x,dt_fmt) for x in time_interval]
    print("Time interval from",new_date_range[0]," to ",new_date_range[1])
    
    

    fig=plt.figure(figsize=figsize)
    #plt.title("start time "+date_range[0])
    fig.subplots_adjust(hspace=0.09)
    
    # RPW
    ax = plt.subplot(2,1,1)
    plt.title("start time "+new_date_range[0])
    rpw_plot_psd(rpw_psd,xlabel=False,colorbar=cbars,frequency_range=rpw_freq_range,cmap=rpw_cmap)
    if(invert_rpw_axis):
        plt.gca().invert_yaxis()
    plt.xlim(*time_interval)
    for mk in markers:
        ax.axvline(dt.datetime.strptime(mk,dt_fmt),c=markers[mk],lw=markerwidth)
    for ev in events:
        if ev.paint_in in ["both","rpw"]:
            ev.paint()


    #STIX
    ax2 = plt.subplot(2,1,2)
    _=stix_plot_spectrogram(stx_cts,colorbar=cbars,energy_range=stix_energy_range,cmap=stix_cmap,logscale=stix_logscale)
    for mk in markers:
        _.axvline(dt.datetime.strptime(mk,dt_fmt),c=markers[mk],lw=markerwidth)
    for ev in events:
        if ev.paint_in in ["both","stix"]:
            ev.paint()
    plt.xlim(*time_interval)

    
    
    
### RPW frequency drift

#format model for fit functions
# param 1: factor
# param 2: time shift (avg of distribution)
# param 3: width of distribution
# param 4: additive (free)
def fit_func_gaussian(x,a,b,c,d):
    return (10**a) * np.exp(-(x-b)**2/( c**2)) + (10**d)
    
def estimate_rmse(x,y,model,params):
    rmse = 0
    for i in range(len(x)):
        rmse += (y-model(x,*params))**2
        
    return np.sqrt(np.mean(rmse))

def dt_to_sec_t0(time_data,t0=None):
    if(t0==None):
        t0=time_data[0]
    time_dts = [time_data[i]-t0 for i in range(len(time_data))]
    t_0 = np.array([t.seconds + t.microseconds/1e6 for t in time_dts])
    
    return [t_0,time_data[0]]

def sec_t0_to_dt(secs, t0):
    tmes = [t0 + dt.timedelta(seconds=secs[j]) for j in range(len(secs))]
    return tmes
    
def rpw_fit_freq_peaks(rpw_psd,peak_model,date_range,frequency_range=None,initial_pos_guess=None,excluded_freqs=[],dt_fmt="%d-%b-%Y %H:%M:%S"):
    time_data = rpw_psd["time"]
    v_data = rpw_psd["v"]
    freq_data = rpw_psd["frequency"]
    
    
    #estimate error
    t_err = np.mean(time_data[1:]-time_data[:-1]).seconds/2.
    f_err = (freq_data[1:]-freq_data[:-1])/2.
    print("Estimated uncertainty:")
    print("  Time: {} s".format(t_err))
    print("  Freq: between {}-{} kHz".format(np.min(f_err),np.max(f_err)))
    print("Defining time reference...")
    # selecting fit intervals
    date_range_dt = [dt.datetime.strptime(x,dt_fmt) for x  in date_range]

    idx_sel_time = np.logical_and(time_data<=date_range_dt[1],time_data>=date_range_dt[0])
    
    time_data = time_data[idx_sel_time]
    v_data = v_data[:,idx_sel_time]
    
    if(frequency_range):
        idx_sel_freq = np.logical_and(freq_data<=frequency_range[1],freq_data>=frequency_range[0])
        freq_data = freq_data[idx_sel_freq]
        v_data = v_data[idx_sel_freq,:]
        
    # convering time to seconds
    time_sec,t0 = dt_to_sec_t0(time_data)
    time_span = time_sec[-1]-time_sec[0] 
    print(" t0 = ",dt.datetime.strftime(t0,dt_fmt))
    print("Fitting peaks for {} frequencies between {} kHz and {} kHz".format(len(freq_data),freq_data[0],freq_data[-1]))
    curve_fits = {}
    curve_fits_meta = {}
    #asume that peak of max freq. is close to the beggining of the timespan
    prev_center = None
    if(initial_pos_guess):
        prev_center = dt_to_sec_t0([dt.datetime.strptime(pos_guess,dt_fmt)],t0)[0][0]
    # starting point (time,freq)
    starting_point =[]
    for i in range(len(freq_data)-1,-1,-1): 
        
        if(freq_data[i] in excluded_freqs):
            print("[{}] {:.0f} kHz   : Excluded!   omitted.".format(i,freq_data[i]))
            continue
        
        
        #if not defined, use approx position in timespan (lineal)
        if(not prev_center):
            prev_center= time_sec[0]    
        
             
        x_ = time_sec
        y_ = v_data[i,:]  #V[freqn,date_idx]
        curve_fits_meta={
                         "t":x_,
                         "y":y_,
                         "t0":t0,
                         "dt":t_err,
                         "time_interval":date_range,
                         "excluded_f":excluded_freqs
                        }
        #fit_bounds = [(1e-18,1e-11),(0,np.max(t_sec0)),(1,1000),(1e-18,1)]
        #fit_bounds = ((-18.,0.,1.,-18.),(-14.,np.max(x_),time_span,-14.))
        try:
            #if(p0):
            
            init_guess = [np.log10(np.max(y_)),prev_center,60.,-16.]
            popt,pcov = cfit(peak_model,x_,y_,p0=init_guess, method="lm")#,bounds=fit_bounds)
       
            # for cases where aprameters were found
            if (len(popt)>0) :
                # diference with previous found point
                dif = popt[1]-prev_center
                if(i==len(freq_data)-1):
                    dif = 0
                # discard if center out of bounds
                if(popt[1]<x_[0]*0.8 or popt[1]>x_[-1]*1.2):
                    print("[{}] {:.0f} kHz   : Not in bounds! omitted.".format(i,freq_data[i]))
                    #popt= []
                    #pcov = []
                else:
                    if(len(starting_point)==0):
                        starting_point = [freq_data[i],sec_t0_to_dt([popt[1]],t0)[0]]
                        sp_t=dt.datetime.strftime(starting_point[1],dt_fmt)
                        #curve_fits_meta={
                         #"t":x_,
                         #"y":y_,
                         #"t0":t0,
                         #"dt":t_err,
                         #"df":f_err[i],
                         #"time_interval":date_range,
                         #"excluded_f":excluded_freqs
                        #}
                        print("Starting point ---------- frequency: {:.0f}+-{:.0f} kHz   time: {}".format(freq_data[i],f_err[i],sp_t))
                    
                    
                    # FITTING ERROR
                    rmse = estimate_rmse(x_,y_,peak_model,popt)
                    snr = 10**(popt[0]-popt[3])
                    
                    ## ADD CURVE FIT TO SOLUTIONS
                    curve_fits[freq_data[i]] = {
                     "params":popt,
                     "covar":pcov,
                     "rmse":rmse,
                     "snr":snr,
                     "df":f_err[i]}
                    
                    
                    print("[{}] {:.0f} kHz: Fit found!   t-t0: {:.2f} s   Dif.: {:.2f} s  Log10(RMSE): {:.2f}  Log10(S/N): {:.2f}".format(i,freq_data[i],popt[1],dif,np.log10(rmse),np.log10(snr)))
                    
                    prev_center = popt[1]#np.mean([time_span * (1-i/len(freq_data))**2,popt[1]])
            
        except:
            print("[{}] {:.0f} kHz   : Not found".format(i,freq_data[i]))
            #popt= []
            #pcov=[]
        
    #print(curve_fits.keys()) 
    fit_results ={
        "freq_fits":curve_fits,
        "metadata":curve_fits_meta
    }
    
    return fit_results
def rpw_plot_fit_results(fit_results,rpw_psd,cmap="jet",fit_limits=False):
    dt_fmt="%d-%b-%Y %H:%M:%S"
    cmap = mpl.cm.get_cmap(cmap)
    
    c_fits=fit_results["freq_fits"]
    meta = fit_results["metadata"]
    
    flist = list(c_fits.keys())
    if(fit_limits):
        frequency_range=[int(flist[0]),int(flist[-1])]
        dt_range=sec_t0_to_dt(meta["t"],meta["t0"])
        dt_range = [dt_range[0]-dt.timedelta(seconds=20),dt_range[-1]+dt.timedelta(seconds=20)]
        dt_range = [dt.datetime.strftime(x,dt_fmt)for x in dt_range]
        
        rpw_plot_psd(rpw_psd,cmap="binary")#,frequency_range=frequency_range)
        plt.gca().invert_yaxis()
        solar_event(event_type="interval",times={'start':dt_range[0],'end':dt_range[1]},color="blue",linewidth=0.5,hl_alpha=0.2).paint()
        #plt.xlim(dt_range)
    else:
        rpw_plot_psd(rpw_psd,cmap="binary")
    #print("frnge",frequency_range)
    
    for i in range(len(flist)):
        695700
        params = c_fits[flist[i]]["params"]
        covars = c_fits[flist[i]]["covar"]
        if(len(params)>0):
            
            t0 = meta["t0"]
            ctime = sec_t0_to_dt([params[1]],t0=t0)[0]
            times_sigma1= sec_t0_to_dt([params[1]-params[2],params[1]+params[2]],t0=t0)
            times_sigma1 = times_sigma1[1]-times_sigma1[0]
            ydat = meta["y"]
            f_sigma = c_fits[flist[i]]["df"]
        
            rgba = cmap(i/len(flist))
            #delays.append(c_fits[flist[i]]["params"][1])
            #freqs.append(flist[i])
            lbl = "{} MHz".format(round(flist[i]/1000.,2))
            if(len(flist)>20 and i%3!=0 and i!=len(flist)-1):
                lbl=None
            plt.errorbar(ctime,int(flist[i]),xerr=np.abs(times_sigma1),yerr=np.abs(f_sigma),color=rgba,
                         label=lbl,marker="o",markersize=3)
    plt.legend(ncol=3,fontsize=9)
    plt.xlabel("Date")
    plt.ylabel("Frequency [MHz]")
    myFmt = mdates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(myFmt)
    
    

def rpw_freq_drifts(fit_results,excluded_freqs=[],):
    peak_fits =fit_results["freq_fits"]
    meta = fit_results["metadata"]
    flist = np.array(list(peak_fits.keys()))
    # peak times
    delays = []
    # frequencies
    freqs = []
    # peak time unceertainty
    devs = []
    # freq uncertainty
    dfs = []
    #time uncertainty695700
    dts = []
    
    
    for i in range(len(flist)):
        params_ =peak_fits[flist[i]]["params"]
        covs_ =peak_fits[flist[i]]["covar"]
        if(len(params_)>0 and not int(flist[i]) in excluded_freqs ):
            delays.append(params_[1])
            freqs.append(flist[i])
            #error
            #devs.append(np.sqrt(np.diag(covs_)[1]))
            dts.append(np.mean(meta["dt"])/2.)
            devs.append(np.sqrt(np.abs(params_[2])))#(2.*np.mean(peak_fits[flist[i]]["dt"])))

            dfs.append(peak_fits[flist[i]]["df"])
            #plt.scatter(int(flist[i]),cf[flist[i]]["params"][1], label=i)
    
    delays = np.array(delays)
    devs = np.abs(np.array(devs))
    dfs = np.abs(np.array(dfs))
    freqs = np.array(freqs)
   
    #f drift estimationsololab.
    dif_freqs = freqs[1:]-freqs[:-1]
    dif_delays = delays[1:]-delays[:-1]
    f_drifts = dif_freqs/dif_delays

    err_delays = devs[:-1]#np.sqrt((devs[1:]**2) + (devs[:-1]**2))
    err_freqs =  dfs[:-1]#np.sqrt((dfs[1:]**2) + (dfs[:-1]**2))
    
    #print(err_delays[:],dif_delays[:],err_freqs[:],dif_freqs[:])
    err_fdrift = np.abs( f_drifts[:]*np.sqrt((err_delays[:]/dif_delays[:])**2 + (err_freqs[:]/dif_freqs[:])**2) )

    return_dict = {"frequencies":flist,
                   "conv_frequencies": freqs,
                   "delays":delays,
                   "freq_drifts":f_drifts,
                   "sigma_dfdt":err_fdrift,
                   "sigma_f" :err_freqs,
                   "sigma_tpeak":err_delays,
                   "sigma_t":dts}
    #print([(len(return_dict[x]),x) for x in list(return_dict.keys())])
    
    #print(f_drifts/1000)
    return  return_dict
def rpw_plot_freq_drift(freq_drifts,errorbars=False,limit_cases=True):
    
    freqs = freq_drifts["conv_frequencies"]
    f_drifts = freq_drifts["freq_drifts"]
    
    maxyerr=np.mean([t for t in freq_drifts["sigma_dfdt"] if np.abs(t)!=np.inf ])
    ax = plt.gca()
    interval_centers=(freqs[1:]+freqs[:-1])/2
    fdrifts_mhz = f_drifts/1000. #in MHz s-1
    for i in range(len(fdrifts_mhz)):
        col = "red" if fdrifts_mhz[i]<0 else "blue"
        yerr=freq_drifts["sigma_dfdt"][i]/1000. #in MHz s-1
        
        yerr = yerr if yerr!=np.inf else maxyerr/1000.
        if(limit_cases and yerr/np.abs(fdrifts_mhz[i])>=1):
            continue
        #print(yerr/np.abs(fdrifts_mhz[i]))
        
        
        ax.scatter(interval_centers[i],np.abs(fdrifts_mhz[i]),c=col,s=8)
        
        bar_alpha=0.3
        if(errorbars=="both"):
            markers, caps, bars = ax.errorbar(interval_centers[i],np.abs(fdrifts_mhz[i]),xerr=freq_drifts["sigma_f"][i],yerr=yerr,c=col,markersize=3,marker="o")
            [bar.set_alpha(bar_alpha) for bar in bars]
        elif(errorbars=="x"):
            markers, caps, bars = ax.errorbar(interval_centers[i],np.abs(fdrifts_mhz[i]),xerr=freq_drifts["sigma_f"][i],c=col,markersize=3,marker="o")
            [bar.set_alpha(bar_alpha) for bar in bars]
        elif(errorbars=="y"):
            markers, caps, bars = ax.errorbar(interval_centers[i],np.abs(fdrifts_mhz[i]),rerr=yerr,c=col,markersize=3,marker="o")
            [bar.set_alpha(bar_alpha) for bar in bars]
            
            
    
    neg_patch = mpatches.Patch(color="red", label='df/dt < 0')
    pos_patch = mpatches.Patch(color="blue", label='df/dt > 0')
    plt.legend(handles=[neg_patch,pos_patch],fontsize=8)
    
   # ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
    
    #for i in range(len(interval_centers)):
        #plt.text(interval_centers[i],fdrifts_mhz[i],str(int(interval_centers[i]))+" MHz",
        #         fontsize=10,horizontalalignment="center",verticalalignment="bottom")
    #plt.ylim(1,5)
    plt.yscale("log")
    plt.xscale("log")#,subs=[1,2,3,4,5,6,7,8,9])
    plt.xlabel("Frequency [kHz]")
    plt.ylabel("Frequency drift rate  [MHz sec$^{-1}$]")
    

def rpw_plot_fit_summary(rpw_psd,fit_results,freq_drifts,fit_limits=True,savepath=None,grid=True,errorbars=False):

    curve_fits = fit_results["curve_fits"]
    cf_meta = fit_results["metadata"]
    fit_interval = cf_meta["time_interval"]
    
    fig=plt.figure(figsize=(16,4),dpi=120)
    spec3 = gridspec.GridSpec(ncols=4, nrows=1)

    fig.add_subplot(spec3[0, 1:])
    rpw_plot_fit_results(curve_fits,rpw_psd,fit_limits=fit_limits)
    
    interv_times = curve_fits
    
    solar_event(event_type="interval",times={'start':fit_interval[0],'end':fit_interval[1]},hl_alpha=0.4).paint()
    fig.add_subplot(spec3[0,0])
    if(grid):
        plt.grid(which='both',color="#EEEEEE")
        plt.grid(color="#CCCCCC")
    rpw_plot_freq_drift(freq_drifts,errorbars)

    
    
    fig.tight_layout()
    if(savepath):
        plt.savefig(savepath,bbox_inches='tight')
        
    
# receivesfrequenciesin kHz
# returns plasma number density in cm-3
def ne_from_freq(freqs,coef=9.):
    return [(f/coef)**2 for f in freqs]

def freq_from_ne(ne,coef=9.):
    return [coef*np.sqrt(n) for n in ne]

def dfdn_from_ne(ne,coef=9.):
    return[(coef/2)*(1./np.sqrt(n)) for n in ne]
def ne_from_r_leblanc(radius):
    return [ 3.3e5*(r**(-2))+4.1e6*(r**(-4))+8.0e7*(r**(-6)) for r in radius]

def r_from_ne(nes,ne_model,r_interv=[1,400],n_iter=5,c=0.1,npoints=1000,error=False):
    
    r_mins = []
    r_err= []
    for  n in nes:
        bounds = r_interv.copy()
        

        for i in range(n_iter):
            
            r_span = np.linspace(bounds[0],bounds[1],npoints)
            r_span_l = bounds[1]-bounds[0]
            ne_span = ne_model(r_span)
            
            r_min = r_span[np.argmin(np.abs(np.array(ne_span)-n))]
            bounds =[max(r_min-c*r_span_l,r_interv[0]),min(r_min+c*r_span_l,r_interv[1])]
        r_err.append(bounds[1]-bounds[0])
        r_mins.append(r_min)
    if(error):
        return r_mins,r_err
    return r_mins
def r_from_freq(freqs,ne_model):
    return r_from_ne(ne_from_freq(freqs),ne_model)

def freq_from_r(r,ne_model):
    return freq_from_ne(ne_model(r))
def dndr_from_r(radius):
    return [ -6.6e5*(r**(-3))-16.4e6*(r**(-5))-48.0e7*(r**(-7)) for r in radius]

def convert_RoSec_to_c(vels):
    #c = 299792.458
    #conv_fact=695700  #695700km = 1 R0
    
    return [(v*km_per_Rs)/speed_c_kms for v in vels]

def convert_c_to_roSec(vels):
    #c = 299792.458
    #conv_fact= 695700  #695700km = 1 R0
    return [(v*speed_c_kms)/km_per_Rs for v in vels]


def rpw_estimate_beam_velocity(freq_drifts,density_model,r_interv=[0.1,300],n_iter=5,c=0.01,npoints=1000,weight_v_error=1.,only_neg_drifts=True):
    freqs = freq_drifts["conv_frequencies"]
    freqs_low_bound = freqs[:-1]
    freqs = (freqs[1:]+freqs[:-1])/2.
    

    delays = freq_drifts["delays"]
    delays = (delays[1:]+delays[:-1])/2.
    
    dfdt = freq_drifts["freq_drifts"]
    
    dt = freq_drifts["sigma_t"]

    if(only_neg_drifts):
        iidx = dfdt<0
        dfdt = dfdt[iidx]
        freqs = freqs[iidx]
        freqs_low_bound = freqs_low_bound[iidx]
        delays = delays[iidx]
    
    
    
    n_e = ne_from_freq(freqs)
    #print(len(n_e),len(freq_drifts["sigma_f"][:len(freqs)]),len(freqs))
    err_ne = n_e[:]*((2/9)*freq_drifts["sigma_f"][:len(freqs)]/freqs[:])#*((1/6)*(freq_drifts["sigma_f"][:len(freqs)]/freqs[:]))
    
    
    rads,err_r = r_from_ne(n_e,density_model,r_interv=r_interv,n_iter=n_iter,c=c,npoints=npoints,error=True)
    err_r = err_r[:] + np.array(rads[:])*(err_ne[:]/n_e[:])
    
    dfdn=dfdn_from_ne(n_e)
    err_dfdn = dfdn[:]*(freq_drifts["sigma_f"][:len(freqs)]/freqs[:])*np.sqrt(1+(1/36))
    
    dndr=dndr_from_r(rads)
    err_dndr = np.abs(dndr[:]*(err_ne[:]/n_e[:])*np.sqrt(1+(1/(2*3.3e5+4*4.1e6+6*8.0e7)**2)))
    
    drdt = []
    drdt_err=[]
    for i in range(len(dfdt)):
        if(dfdt[i]>0):
            continue
        v_trig = dfdt[i]*(dndr[i] **(-1) )*( dfdn[i]**(-1) )
        #print(dfdt[i],dndr[i] ,dfdn[i],v_trig)
        err_v = v_trig*np.sqrt((freq_drifts["sigma_dfdt"][i]/dfdt[i])**2 + (err_dfdn[i]/dfdn[i])**2 + (err_dndr[i]/dndr[i])**2)
        
        drdt.append( v_trig )
        drdt_err.append( err_v*weight_v_error)
        
        
    return_dict = {
        "frequencies":freq_drifts["conv_frequencies"],
        "freq_average":freqs,
        "freq_low_bound":freqs_low_bound,
        "delays":delays,
        "n_e":n_e,
        "r":rads,
        "dfdt":dfdt,
        "drdt":drdt,
        "dndr":dndr,
        "dfdn":dfdn,
        "err_drdt":drdt_err,
        "err_dndr":err_dndr,
        "err_n_e":err_ne,
        "err_r":err_r,
        "dt":dt[:len(freqs)]
        
    }
    #print([(len(return_dict[x]),x) for x in list(return_dict.keys())])
    
    return return_dict
    
    
    
def rpw_plot_typeIII_diagnostics(rpw_psd,fit_results,freq_drifts,trigger_velocity,figsize=(16,15),dpi=150,errorbars="both",dfdt_errorbars="both",grid=True,fit_limits=False,cmap="jet"):
    
   # print(freq_drifts["conv_frequencies"])

    peak_fits=fit_results["freq_fits"]
    pf_meta = fit_results["metadata"]
    fit_interval = pf_meta["time_interval"]
    
    t0 = pf_meta["t0"]
    
    timeax = sec_t0_to_dt(trigger_velocity["delays"],t0)
    
    # TIME error
    delta_t = trigger_velocity["dt"]
    delta_t_dt = [dt.timedelta(seconds=t) for t in delta_t]
    
    terr = freq_drifts["sigma_tpeak"]
    terr_dt=[dt.timedelta(seconds=t) for t in terr]
    
    r_err = trigger_velocity["err_r"]
    f_err = freq_drifts["sigma_f"]
    

    cmap = mpl.cm.get_cmap(cmap)
    # create figure and grid
    fig=plt.figure(figsize=figsize,dpi=dpi)
    spec3 = gridspec.GridSpec(ncols=4, nrows=9,wspace=0.4,hspace=0.)
    
    # PLOT SPECTROGRAM
    ax=fig.add_subplot(spec3[:2, 1:])
    
    rpw_plot_fit_results(fit_results,rpw_psd,fit_limits=fit_limits)
    plt.gca().invert_yaxis()
    solar_event(event_type="interval",times={'start':fit_interval[0],'end':fit_interval[1]},hl_alpha=0.4,color="#AADDAA").paint()
    ax.xaxis.set_label_position('top')
    #ax.xaxis.tick_top()
    plt.xlabel("start time: {}".format(dt.datetime.strftime(rpw_psd["time"][0],"%d-%b-%Y %H:%M:%S")))
    #PLOT FREQ. DRIFTS
    ax=fig.add_subplot(spec3[:2,0])
    if(grid):
        plt.grid(which='both',color="#EEEEEE")
        plt.grid(color="#CCCCCC")
    rpw_plot_freq_drift(freq_drifts,errorbars)
    ax.xaxis.set_label_position('top')
    #ax.xaxis.tick_top()
    
    
    # VELOCITY DIAGNOSTICS
    vels = convert_RoSec_to_c(trigger_velocity["drdt"])
    err_vels = convert_RoSec_to_c(trigger_velocity["err_drdt"])
    
    
    
    # PLOT DIAGNOSTICS VS TIME
    ax=fig.add_subplot(spec3[3:4,:2])
    
    
    #select in range velocities (physical datapoints)
    
    selected_datapoints = [x for x in range(len(vels)) if (vels[x]>0 and vels[x]<=1) ]
    mean_selected = np.mean([vels[x] for x in selected_datapoints])
    
    maxs = [vels[i]+err_vels[i] for i in selected_datapoints]
    mins =[max(vels[i]-err_vels[i],1e-10) for i in selected_datapoints]
    bot_avg = np.mean(mins)
    top_avg = np.mean(maxs)

    
    #ax.axhline(top_avg,c="grey",linestyle="--")
    #ax.axhline(bot_avg,c="grey",linestyle="--")
    
    plt.axhspan(bot_avg, top_avg, color="lightgrey", alpha=0.5)
    
    ax.axhline(mean_selected,c="k",linestyle="--",label="average = {:.2f} c".format(mean_selected))
    #ax.axhline(1,c="r",linestyle="--",label="speed of light")
   
    ax.set_xlabel("$t-t_0$ [sec] ",fontsize=13)
    ax.set_ylabel("v/c",fontsize=13)
    
    
    for f_i in range(len(trigger_velocity["freq_average"])):
        if(vels[f_i]<0 or vels[f_i]>1):
            err_vels[f_i]=0
        rgba = cmap((f_i+1)/len(trigger_velocity["freq_average"]))
        ax.scatter(trigger_velocity["delays"][f_i],vels[f_i],color=rgba,marker="o",s=20)
        ax.errorbar(trigger_velocity["delays"][f_i],vels[f_i],xerr=terr[f_i]+delta_t[f_i],yerr=np.abs(err_vels[f_i]),c=rgba,markersize=2)
    
    
    ax.xaxis.tick_top()
    #plt.ylim(1e-3,10)
    bot_lim = np.min(mins)/2. if np.min(mins) else 1e-3
    bot_lim = max(bot_lim,1e-3)
    plt.ylim(bot_lim,1)
    ax.set_yscale("log")
    ax.xaxis.set_label_position('top')
    xmin, xmax = ax.get_xlim()
    xmin,xmax = sec_t0_to_dt([xmin,xmax],t0)
    
    #ax.legend()
    
    
    ax2=fig.add_subplot(spec3[4:5,:2])
    ne_err = trigger_velocity["err_n_e"]
    for f_i in range(len(trigger_velocity["freq_average"])):
        rgba = cmap((f_i+1)/len(trigger_velocity["freq_average"]))
        ax2.scatter(timeax[f_i],trigger_velocity["n_e"][f_i],color=rgba,marker="o",s=20)
        ax2.errorbar(timeax[f_i],trigger_velocity["n_e"][f_i],yerr=np.abs(ne_err[f_i]),xerr=terr_dt[f_i]+delta_t_dt[f_i],c=rgba)
    ax2.set_yscale("log")
    ax2.set_ylabel("$n_e$ [cm$^{-3}$]",fontsize=13)
    ax2.set_xticks([])
    plt.xlim(xmin,xmax)
    
    fig.add_subplot(spec3[5:6,:2])
    for f_i in range(len(trigger_velocity["freq_average"])):
        rgba = cmap((f_i+1)/len(trigger_velocity["freq_average"]))
        plt.scatter(timeax[f_i],trigger_velocity["r"][f_i],color=rgba,marker="o",s=20,label="{} kHz".format(int(trigger_velocity["frequencies"][f_i])))
        plt.errorbar(timeax[f_i],trigger_velocity["r"][f_i],xerr=terr_dt[f_i]+delta_t_dt[f_i],yerr=np.abs(r_err[f_i]),c=rgba,markersize=2)
    plt.xlabel("Time (UT)  $t_0$ = {}".format(dt.datetime.strftime(t0,std_date_fmt)),fontsize=13)
    plt.ylabel("r $[R_o]$",fontsize=13)
    plt.xticks()
    myFmt = mdates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(myFmt)
    #plt.legend(fontsize=7,ncol=4)
    plt.xlim(xmin,xmax)
    
    
    # PLOT DIAGNOSTICS VS R
 
    ax=fig.add_subplot(spec3[3:4,2:])
    for f_i in range(len(trigger_velocity["freq_average"])):
        rgba = cmap((f_i+1)/len(trigger_velocity["freq_average"]))
        r_AU = trigger_velocity["r"][f_i]*(1/Rs_per_AU)
        r_err_AU = r_err[f_i]*(1/Rs_per_AU)
        if(vels[f_i]<0):
            err_vels[f_i]=0
        ax.errorbar(r_AU,vels[f_i],yerr=np.abs(err_vels[f_i]),xerr=np.abs(r_err_AU),c=rgba,marker="o",markersize=2)
        
        ax.scatter(r_AU,vels[f_i],color=rgba,marker="o",s=20)
    plt.yscale("log")
    
    plt.axhspan(bot_avg, top_avg, color="lightgrey", alpha=0.5)
    
    ax.axhline(mean_selected,c="k",linestyle="--",label="average = {:.2f} c".format(mean_selected))
    #ax.axhline(1,c="r",linestyle="--",label="speed of light")
   
    
    
    plt.xlabel("r $[AU]$",fontsize=13)
    plt.ylabel("v/c",fontsize=13)
    plt.ylim(max(1e-3,np.min(mins)/2.),1)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    ax.yaxis.tick_right()

    #ax.yaxis.set_label_position('right')
    
    
    plt.legend(fontsize=13)
    
    ax2=fig.add_subplot(spec3[4:5,2:])
    for f_i in range(len(trigger_velocity["freq_average"])):
        rgba = cmap((f_i+1)/len(trigger_velocity["freq_average"]))
        ax2.errorbar(trigger_velocity["r"][f_i],trigger_velocity["n_e"][f_i],yerr=ne_err[f_i],xerr=r_err[f_i],c=rgba)
        ax2.scatter(trigger_velocity["r"][f_i],trigger_velocity["n_e"][f_i],color=rgba,marker="o",s=20)
    plt.yscale("log")

    plt.ylabel("$n_e$ [cm$^{-3}$]",fontsize=13)
    ax2.yaxis.tick_right()
    ax2.set_xticks([])
    #ax2.yaxis.set_label_position('right')
    

    #print(trigger_velocity["freq_average"])
    ax3=fig.add_subplot(spec3[5:6,2:],)
    for f_i in range(len(trigger_velocity["freq_low_bound"])):
        #print(freq_drifts["conv_frequencies"][f_i])
        rgba = cmap((f_i+1)/len(trigger_velocity["freq_low_bound"]))
        lbl = "{} MHz".format(round(trigger_velocity["freq_low_bound"][f_i]/1000.,2))
        if(len(trigger_velocity["freq_average"])>15 and f_i%2!=0 and f_i!=len(trigger_velocity["freq_average"])-1):
                lbl=None
        plt.scatter(trigger_velocity["r"][f_i],trigger_velocity["freq_average"][f_i],color=rgba,marker="o",s=20,label=lbl)
        ax3.errorbar(trigger_velocity["r"][f_i],trigger_velocity["freq_average"][f_i],xerr=r_err[f_i],yerr=f_err[f_i],c=rgba)
    plt.xlabel("r $[R_o]$",fontsize=13)
    plt.ylabel("Frequency [Hz]",fontsize=13)
    plt.yscale("log")
    plt.legend(fontsize=9,ncol=3)
    #plt.yscale("log")
    #plt.xscale("log")
    ax3.yaxis.tick_right()

    #ax3.yaxis.set_label_position('right')
    


    




 