#!/home/anja/anaconda3/bin/python
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import pickle,glob,re
from netCDF4 import Dataset
import altair as alt
import matplotlib.patches as mpatches
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime,timedelta
from matplotlib.dates import DateFormatter, HourLocator
def readSSH(fname):	

    df = pd.read_csv(fname,delimiter='\t')	
    df.columns = ['date', 'ssh']	
    df['ssh'] = pd.to_numeric(df['ssh'])	
    df['date'] = pd.to_datetime(df['date'],format='%d.%m.%Y %H:%M') - dt.timedelta(hours=1)		
    df = df.set_index('date')	

    return df

def readVeter(fname):	

    df = pd.read_csv(fname,delimiter='\t')	
    df.columns = ['date', 'speed','dir']	
    df['speed'] = pd.to_numeric(df['speed'])	
    df['date'] = pd.to_datetime(df['date'],format='%d.%m.%Y %H:%M') - dt.timedelta(hours=1)		
    df = df.set_index('date')	

    return df

def readMsl(fname):	

    df = pd.read_csv(fname,delimiter='\t')	
    df.columns = ['date', 'msl']	
    df['msl'] = pd.to_numeric(df['msl'])	
    df['date'] = pd.to_datetime(df['date'],format='%d.%m.%Y %H:%M') - dt.timedelta(hours=1)		
    df = df.set_index('date')	

    return df


def read_nc_var(fname,var):
    # print "read_nc_var() reading file:"
    # print fname
    ncfile=Dataset(fname,'r')
    return(np.squeeze(ncfile.variables[var][:]))

def read_nc_datetimes(ncfile,time_varname):
    ncid=Dataset(ncfile,'r')
    t=ncid.variables[time_varname]
    t0 = t.units
    mt =re.findall(r'\d{4}-\d+-\d+',t0)

    return np.array([datetime.strptime(mt[0],'%Y-%m-%d') + timedelta(seconds=i) for i in t[:]])

def read_ec_datetimes(ncfile,time_varname):
    ncid=Dataset(ncfile,'r')
    t=ncid.variables[time_varname]
    t0 = t.units
    mt =re.findall(r'\d{4}-\d+-\d+',t0)

    return np.array([datetime.strptime(mt[0],'%Y-%m-%d') + timedelta(hours=i) for i in t[:]])

def readFile(fname):   
    ensNum = re.findall('\_(\d\d)\.',fname)[0]
    df = pd.DataFrame(columns=['dates',str(ensNum)])
  
    sossheig = read_nc_var(fname,'zos')
    time = read_nc_datetimes(fname,'time_counter')	
  
    df['dates'] = time
    df[str(ensNum)] = sossheig
    df = df.set_index(['dates'])
    
    return df

def readECFile(fname,param):   
    ensNum = re.findall('\_(\d\d)\.',fname)[0]
    df = pd.DataFrame(columns=['dates',str(ensNum)])
    lon0 = 13.7
    lat0 = 45.55

    lon = read_nc_var(fname,'lon') 
    lat = read_nc_var(fname,'lat') 
    i,j = get_latlonIndex(lon0,lat0,lon,lat)
    u10 = read_nc_var(fname,'u10')[:,j,i]
    v10 = read_nc_var(fname,'v10')[:,j,i] 
    msl = read_nc_var(fname,'msl')[:,j,i]
    time = read_ec_datetimes(fname,'time')	
  
    
    df['dates'] = time
    if 'msl' in param:
        df[str(ensNum)] = msl*0.01
    else:
        df[str(ensNum)] = np.sqrt(u10*u10 + v10*v10)
    df = df.set_index(['dates'])
    
    return df

def readECMWF(datadir,param):
    ens_df = pd.DataFrame()
    for file in glob.glob(datadir + '/ecmwf_*.nc'):
        df = readECFile(file,param)     

        ens_df = pd.concat([ens_df,df],axis=1,join='outer')
    return ens_df

# def readEns(pickle_file):
#     f = open(pickle_file,'rb')
#     run,tide_run = pickle.load(f)
#     f.close()
#     return f

def readEns(datadir):
    ens_df = pd.DataFrame()
    for file in glob.glob(datadir + '/ssh_kp_*.nc'):
        df = readFile(file)     

        ens_df = pd.concat([ens_df,df],axis=1,join='outer')
    return ens_df

def calcMeanEns(data,dh):

	dif = data + dh
	
	mean_ens = dif.dropna().mean(axis=1)	
	max_ens = dif.dropna().max(axis=1)  
	min_ens = dif.dropna().min(axis=1) 
	median_ens = dif.dropna().median(axis=1)
	mean_dates = dif.dropna().index	
	
	predicted = pd.DataFrame()
	predicted['ssh'] = mean_ens
	predicted['dates'] = mean_dates			
	predicted['max'] = max_ens 
	predicted['min'] = min_ens
	predicted['median'] = median_ens
	
	predicted = predicted.set_index(['dates'])	

	return predicted

def calcMeanECMWF(data):
		
	mean_ens = data.dropna().mean(axis=1)	
	max_ens = data.dropna().max(axis=1)  
	min_ens = data.dropna().min(axis=1) 
	median_ens = data.dropna().median(axis=1)
	mean_dates = data.dropna().index	
	
	predicted = pd.DataFrame()
	predicted['data'] = mean_ens
	predicted['dates'] = mean_dates			
	predicted['max'] = max_ens 
	predicted['min'] = min_ens
	predicted['median'] = median_ens
	
	predicted = predicted.set_index(['dates'])	

	return predicted

def get_latlonIndex(lon0,lat0,lon,lat):

	return np.argmin(np.abs(lon-lon0)),np.argmin(np.abs(lat-lat0))

st.set_page_config(layout="wide")
st.title ("Nemo storm surge ")

#################################################
lnwdth=2
lnwdth1=3
fill_col1='dimgray'
fill_col2='steelblue'
patch_one = mpatches.Patch(color=fill_col1, label='NEMO ens mean')
patch_two = mpatches.Patch(color='darkgrey',label='NEMO ens')
patch_three = mpatches.Patch(color=fill_col2 ,label='NEMO ens min, max')
patch_four = mpatches.Patch(color='k', label='Insitu')	
patch_five = mpatches.Patch(color='skyblue' ,label='NEMO tide')
fsize=12
dateStart=datetime.strptime('20230221', '%Y%m%d')

###################################################

insitu = readSSH('mareografKP_vodostaj.txt')
ens_df = readEns('/home/anja/streamlit_examples/')
veter = readECMWF('/home/anja/streamlit_examples/','veter')
insituVeter= readVeter('/home/anja/streamlit_examples/Koper_wind.txt')
veter_df = calcMeanECMWF(veter)

msl = readECMWF('/home/anja/streamlit_examples/','msl')
insituMsl= readMsl('/home/anja/streamlit_examples/Koper_mslp.txt')
msl_df = calcMeanECMWF(msl)


dateEnd = ens_df.index.max()
insitu_dm = np.nanmean(insitu)

insitu_data = insitu[insitu.index >= dateStart ]	
insituVeter = insituVeter[insituVeter.index >= dateStart]
insituMsl = insituMsl[insituMsl.index >= dateStart]

ens = (ens_df - ens_df.mean()) + insitu_dm

merge_corr = ens.join(insitu_data,how='inner')

dh = - (merge_corr.mean() - merge_corr.mean()['ssh'])
dh = dh.drop(['ssh'])
# dh = dh.drop(['50'])
dh = np.where(np.isnan(dh),0,dh)
# tide = pd.DataFrame(columns=['dates','ssh'])
# tide['ssh'] = ens['50']
# tide['dates'] = ens.index
# tide = tide.set_index(['dates'])
# tide_dif = (tide - tide.mean()) + insitu_dm

# merge_tide = tide_dif.join(insitu_data,how='inner')

# tide_dh = - (merge_tide.mean() - merge_tide.mean()['ssh'])
# tide_dh = tide_dh.drop(['ssh'])
# tide_dh = np.where(np.isnan(tide_dh),0,tide_dh)

# ens = ens.drop(['50'])
# st.header("this is the markdown")
# st.markdown("this is the header")
# st.subheader("this is the subheader")
# st.caption("this is the caption")
# st.code("x=2021")
# st.latex(r''' a+a r^1+a r^2+a r^3 ''')

# st.checkbox('yes')
# st.button('Click')
# st.radio('Pick your gender',['Male','Female'])
# st.selectbox('Pick your gender',['Male','Female'])
# st.multiselect('choose a planet',['Jupiter', 'Mars', 'neptune'])
# st.select_slider('Pick a mark', ['Bad', 'Good', 'Excellent'])
# st.slider('Pick a number', 0,50)

predicted = calcMeanEns(ens, dh)	


# # Plot!
# st.plotly_chart(fig, use_container_width=True)
max_trace = dict(
    x = predicted.index,
    y = predicted['max'],
    mode = 'lines',
    type = 'scatter',
    name = 'max',
    line = dict(shape = 'linear', color = fill_col2, width= lnwdth, dash = 'solid'),
    connectgaps = True
    
)
min_trace = dict(
    x = predicted.index,
    y = predicted['min'],
    mode = 'lines',
    type = 'scatter',
    name = 'min/max NEMO ',
    line = dict(shape = 'linear', color = fill_col2, width= lnwdth, dash = 'solid'),
    connectgaps = True,
    fill='tonexty',
  
)

median_trace = dict(
    x = predicted.index,
    y = predicted['median'],
    mode = 'lines',
    type = 'scatter',
    name = 'median NEMO',
    line = dict(shape = 'linear', color = fill_col1, width= lnwdth1),
    connectgaps = True
)
insitu_trace = dict(
    x = insitu_data.index,
    y = insitu_data['ssh'],
    mode = 'lines',
    type = 'scatter',
    name = 'meritev KP',
    line = dict(shape = 'linear', color = 'black', width= lnwdth1,dash = 'dot'),
    connectgaps = True
)

data = pd.DataFrame()
data['dates'] = [dateStart, dateEnd]
data = data.set_index('dates')
data['res']  = float(300)	

thresh3_trace = dict(
    x = data.index,
    y = data['res'],
    mode = 'lines',
    type = 'scatter', 
    name='300',   
    line = dict(shape = 'linear', color = 'gold',dash = 'dot'),
    connectgaps = True
)

data = pd.DataFrame()
data['dates'] = [dateStart, dateEnd]
data = data.set_index('dates')
data['res']  = float(330)	

thresh4_trace = dict(
    x = data.index,
    y = data['res'],
    mode = 'lines',
    type = 'scatter', 
    name='330',   
    line = dict(shape = 'linear', color = 'orange',dash = 'dot'),
    connectgaps = True
)

data = pd.DataFrame()
data['dates'] = [dateStart, dateEnd]
data = data.set_index('dates')
data['res']  = float(350)	

thresh5_trace = dict(
    x = data.index,
    y = data['res'],
    mode = 'lines',
    type = 'scatter', 
    name='350',   
    line = dict(shape = 'linear', color = 'red',dash = 'dot'),
    connectgaps = True
)



layout =  dict(
    xaxis = dict(title = '2023',titlefont=dict(size=15,color='black')),
    yaxis = dict(title = 'Vodostaj [cm]',titlefont=dict(size=15,color='black')),
    margin = dict(
        l=70,
        r=10,
        b=50,
        t=10
    )
)
data = [max_trace,min_trace,median_trace,insitu_trace,thresh3_trace,thresh4_trace,thresh5_trace]

fig =  go.Figure(data = data, layout=layout)
fig.update_layout(width=1500,height=700,title='Vodostaj v Kopru',titlefont=dict(size=20),margin=dict(
        l=50,
        r=50,
        b=100,
        t=100,
        pad=4
    ))
fig.update_xaxes(showgrid=True,
                 tickformat="%d.%b\n%H",
                 minor=dict(showgrid=True)
)
fig.update_xaxes(gridcolor='gray', griddash='solid', minor_griddash="dot")

for trace in fig['data']: 
    if(trace['name'] == '350'): trace['showlegend'] = False
    if(trace['name'] == '330'): trace['showlegend'] = False
    if(trace['name'] == '300'): trace['showlegend'] = False
    if(trace['name'] == 'max'): trace['showlegend'] = False

fig.update_xaxes(tickfont=dict(family="Arial",
                                 size=15,
                                color='black' ))
fig.update_yaxes(tickfont=dict(family="Arial",
                                 size=15,
                                color='black' ))


fig.update_xaxes(title_font_family="Arial")

st.plotly_chart(fig)


#########################################

max_trace = dict(
    x = veter_df.index,
    y = veter_df['max'],
    mode = 'lines',
    type = 'scatter',
    name = 'max',
    line = dict(shape = 'linear', color = 'lightcoral', width= lnwdth, dash = 'solid'),
    connectgaps = True
)
min_trace = dict(
    x = veter_df.index,
    y = veter_df['min'],
    mode = 'lines',
    type = 'scatter',
    name = 'min/max ECMWF',
    line = dict(shape = 'linear', color = 'lightcoral', width= lnwdth, dash = 'solid'),
    connectgaps = True,
    fill='tonexty'
    
)

median_trace = dict(
    x = veter_df.index,
    y = veter_df['median'],
    mode = 'lines',
    type = 'scatter',
    name = 'median ECMWF',
    line = dict(shape = 'linear', color = 'indianred', width= lnwdth1),
    connectgaps = True
)

insitu_trace = dict(
    x = insituVeter.index,
    y = insituVeter['speed'],
    mode = 'lines',
    type = 'scatter',
    name = 'meritev KP',
    line = dict(shape = 'linear', color = 'black', width= lnwdth1,dash = 'dot'),
    connectgaps = True
)


layout =  dict(
    xaxis = dict(title = '2023',titlefont=dict(size=15,color='black')),
    yaxis = dict(title = 'Hitrost vetra [m/s]',titlefont=dict(size=15,color='black')),
    margin = dict(
        l=70,
        r=10,
        b=50,
        t=10
    )
)
data = [max_trace,min_trace,median_trace,insitu_trace]

fig =  go.Figure(data = data, layout=layout)
fig.update_layout(width=1500,height=700,title='Veter v Kopru',titlefont=dict(size=20),margin=dict(
        l=50,
        r=50,
        b=100,
        t=100,
        pad=4
    ),font=dict(size=18,color='black'))


fig.update_xaxes(showgrid=True,
                 tickformat="%d.%b\n%H",
                 minor=dict(showgrid=True)

)
fig.update_xaxes(gridcolor='gray', griddash='solid', minor_griddash="dot")
for trace in fig['data']:     
    if(trace['name'] == 'max'): trace['showlegend'] = False

fig.update_xaxes(tickfont=dict(family="Arial",
                                 size=15,
                                color='black' ))
fig.update_yaxes(tickfont=dict(family="Arial",
                                 size=15,
                                color='black' ))


fig.update_xaxes(title_font_family="Arial")
st.plotly_chart(fig)
#########################################################33
max_trace = dict(
    x = msl_df.index,
    y = msl_df['max'],
    mode = 'lines',
    type = 'scatter',
    name = 'max',
    line = dict(shape = 'linear', color = 'goldenrod', width= lnwdth, dash = 'solid'),
    connectgaps = True
)
min_trace = dict(
    x = msl_df.index,
    y = msl_df['min'],
    mode = 'lines',
    type = 'scatter',
    name = 'min/max ECMWF',
    line = dict(shape = 'linear', color = 'goldenrod', width= lnwdth, dash = 'solid'),
    connectgaps = True,
    fill='tonexty'
    
)

median_trace = dict(
    x = msl_df.index,
    y = msl_df['median'],
    mode = 'lines',
    type = 'scatter',
    name = 'median ECMWF',
    line = dict(shape = 'linear', color = 'darkgoldenrod', width= lnwdth1),
    connectgaps = True
)

insitu_trace = dict(
    x = insituMsl.index,
    y = insituMsl['msl'],
    mode = 'lines',
    type = 'scatter',
    name = 'meritev KP',
    line = dict(shape = 'linear', color = 'black', width= lnwdth1,dash = 'dot'),
    connectgaps = True
)


layout =  dict(
    xaxis = dict(title = '2023',titlefont=dict(size=15,color='black')),
    yaxis = dict(title = 'Zračni pritisk [mBar]',titlefont=dict(size=15,color='black')),
    margin = dict(
        l=70,
        r=10,
        b=50,
        t=10
    )
)
data = [max_trace,min_trace,median_trace,insitu_trace]

fig =  go.Figure(data = data, layout=layout)
fig.update_layout(width=1500,height=700,title='Zračni pritisk v Kopru',titlefont=dict(size=20),margin=dict(
        l=50,
        r=50,
        b=100,
        t=100,
        pad=4
    ),font=dict(size=18,color='black'))


fig.update_xaxes(showgrid=True,
                 tickformat="%d.%b\n%H",
                 minor=dict(showgrid=True)

)
fig.update_xaxes(gridcolor='gray', griddash='solid', minor_griddash="dot")
for trace in fig['data']:     
    if(trace['name'] == 'max'): trace['showlegend'] = False

fig.update_xaxes(tickfont=dict(family="Arial",
                                 size=15,
                                color='black' ))
fig.update_yaxes(tickfont=dict(family="Arial",
                                 size=15,
                                color='black' ))


fig.update_xaxes(title_font_family="Arial")
st.plotly_chart(fig)
